import numpy as np
import threading
import cv2
import atexit

import time
from pyorbbecsdk import *
import open3d as o3d

# from pyorbbecsdk import Pipeline, Context, Config, OBFormat, OBSensorType, FrameSet, VideoFrame
# from utils_orbbec import frame_to_rgb_image  
from gym_hil.envs.utils_orbbec import frame_to_rgb_image
ESC_KEY = 27

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

class OrbbecCamera:
    def __init__(self, ip="192.168.1.124", port=8090):
        atexit.register(self.stop)

        self.ctx = Context()
        self.device = self.ctx.create_net_device(ip, port)
        if not self.device:
            raise ConnectionError("Failed to create net device")
        
        self.pipeline = Pipeline(self.device)
        self.config = Config()
        self.running = False
        self.latest_frames = {"depth": None, "color": None}
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)  # ← 新增
        self.processing_thread = None
        self.color_profile = None
        self.depth_profile = None
        self.initialize()
        self.start_processing()

    def initialize(self):
        device_info = self.device.get_device_info()
        SUPPORTED_PIDS = {0x080E, 0x0815}  # Gemini 335Le & 435Le
        
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        self.depth_profile = depth_profile_list.get_default_video_stream_profile()
        self.config.enable_stream(self.depth_profile)

        if device_info.get_pid() in SUPPORTED_PIDS:
            self.color_profile = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)\
                .get_video_stream_profile(1280, 800, OBFormat.RGB, 10)
        else:
            self.color_profile = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)\
                .get_video_stream_profile(3840, 2160, OBFormat.H264, 25)
        self.config.enable_stream(self.color_profile)
        # 启动管道
        # self.config.set_align_mode(OBAlignMode.SW_MODE)  # 设置软件对齐模式   /// todo  对齐确认？？
        
        # 创建对齐过滤器（深度对齐到彩色）
        self.align_filter = AlignFilter(OBStreamType.COLOR_STREAM)

        self.pipeline.start(self.config)
        self.pipeline.enable_frame_sync()  # 启用帧同步
    def get_camera_intrinsic(self):
        intrinsics = {}
        depth_intrinsics = self.depth_profile.as_video_stream_profile().get_intrinsic()
        intrinsics['depth_intrinsics'] = depth_intrinsics
        depth_distortion = self.depth_profile.as_video_stream_profile().get_distortion()
        intrinsics['depth_distortion'] = depth_distortion
        color_intrinsics = self.color_profile.as_video_stream_profile().get_intrinsic()
        intrinsics['color_intrinsics'] = color_intrinsics
        color_distortion = self.color_profile.as_video_stream_profile().get_distortion()
        intrinsics['color_distortion'] = color_distortion
        # print("intrinsics is :",intrinsics)
        return intrinsics
    def get_distCoeffs(self):
        intrinsics = self.get_camera_intrinsic()
        if intrinsics is not None:
            temp = intrinsics["color_distortion"]
            color_distortion = [temp.k1, temp.k2, temp.k3, temp.k4, temp.k5, temp.k6, temp.p1, temp.p2]
        self.dist_coeffs = np.asarray(color_distortion)
        return self.dist_coeffs
    def get_intrinsics_mat(self):
        intrinsics = self.get_camera_intrinsic()
        if intrinsics is not None:
            fx = intrinsics["color_intrinsics"].fx
            fy = intrinsics["color_intrinsics"].fy
            cx = intrinsics["color_intrinsics"].cx
            cy = intrinsics["color_intrinsics"].cy
            self.color_intrinsics_mat = np.array([[fx, 0, cx], 
                                                [0, fy, cy], 
                                                [0, 0, 1]])
        return self.color_intrinsics_mat

    def start_processing(self):
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
    
    def _process_frames(self):
        start_time =time.time()
        while self.running:
            try:
                # print(f"{time.time()}proces_frame !!!!!!!!!!!!!")
                frames = self.pipeline.wait_for_frames(100)
                if not frames:
                    continue

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if depth_frame is None or color_frame is None:
                    continue

                # depth_format = depth_frame.get_format()
                # if depth_format != OBFormat.Y16:
                #     print("depth format is not Y16")
                #     return None, None
                
                # aligned_frames = self.align_filter.process(frames)
                
                # aligned_frames = aligned_frames.as_frame_set()
                # color_frame = frames.get_color_frame()
                # depth_frame = frames.get_depth_frame()
                # if not color_frame or not depth_frame:
                #     continue

                width = depth_frame.get_width()
                height = depth_frame.get_height()
                scale = depth_frame.get_depth_scale()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))
                depth_data = depth_data.astype(np.float32) * scale
                depth_data = np.where((depth_data > 10) & (depth_data < 10000), depth_data, 0)
                depth_data = depth_data.astype(np.uint16)
                # print("depth_data is: ",depth_data)
                # depth_data_raw = depth_data.copy()
                # depth_data = temporal_filter.process(depth_data)

                # center_y = int(height / 2)
                # center_x = int(width / 2)
                # center_distance = depth_data[center_y, center_x]
                # current_time = time.time()
                # if current_time - last_print_time >= 1:
                #     print("center distance: ", center_distance)
                #     last_print_time = current_time
                # depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
                # cv2.imshow("Depth Viewer", depth_image)
                # key = cv2.waitKey(1)
                # if key == ord('q') or key == ESC_KEY:
                #     break
                
                # color_image = self._process_color_frame(color_frame)

                # cv2.imshow("color Viewer", color_image)

                # key = cv2.waitKey(1)
                # if key == ord('q') or key == ESC_KEY:
                #     break
                with self.frame_ready:
                    self.latest_frames["depth"] = self._process_depth_frame(depth_frame)
                    self.latest_frames["color"] = self._process_color_frame(color_frame)
                    self.frame_ready.notify_all()

                # with self.lock:
                #     if depth_frame:
                #         self.latest_frames["depth"] = self._process_depth_frame(depth_frame)
                #     if color_frame:
                #         self.latest_frames["color"] = self._process_color_frame(color_frame)
            
            except Exception as e:
                continue
                print(f"Frame processing error: {e}")
    
    def _process_depth_frame(self, frame: VideoFrame):
        # 深度图处理 (Z16格式)
        depth_data = frame.get_data()
        return np.frombuffer(depth_data, dtype=np.uint16).reshape(
            (frame.get_height(), frame.get_width())
        )
    
    def _process_color_frame(self, frame: VideoFrame):
        # 彩色图处理 (使用提供的转换函数)
        return frame_to_rgb_image(frame)

    def get_frame(self,timeout=None):
        """返回最新的深度图和BGR图"""
        # with self.lock:
        #     return {
        #         "depth": self.latest_frames["depth"].copy() if self.latest_frames["depth"] is not None else None,
        #         "color": self.latest_frames["color"].copy() if self.latest_frames["color"] is not None else None
        #     }
        with self.frame_ready:
            while (self.latest_frames["depth"] is None or
                   self.latest_frames["color"] is None):
                if not self.frame_ready.wait(timeout):
                    raise TimeoutError("get_frame timeout")
            return (
                self.latest_frames["color"].copy(),
                self.latest_frames["depth"].copy()
            )

    def stop(self):
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        self.pipeline.stop()
    
    def __enter__(self):
        self.start_processing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

def rgbd2pcd_full_points(rgb, depth, depth_intrinsics_dict):
    # 参数验证
    assert rgb.shape[:2] == depth.shape, "RGB和深度图分辨率不一致"
    height, width = depth.shape
    
    # 准备内参参数
    fx = depth_intrinsics_dict['fx']
    fy = depth_intrinsics_dict['fy']
    cx = depth_intrinsics_dict['cx']
    cy = depth_intrinsics_dict['cy']
    
    # 转换深度图单位 (毫米 -> 米)
    if depth.dtype == np.uint16:
        depth_meters = depth.astype(np.float32) / 1000.0
    else:
        depth_meters = depth.astype(np.float32)
    
    # 创建坐标网格
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)
    
    # 计算3D坐标 (Z = 深度值)
    z = depth_meters
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    
    # 创建点云并赋值
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)  # 形状: (H*W, 3)
    colors = rgb.reshape(-1, 3) / 255.0  # 归一化到[0,1]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print("pcd points shape is", np.asarray(pcd.points).shape)
    
    # # 移除原点附近的无效点 (可选)
    # valid_mask = z.flatten() > 1e-3  # 深度>1mm视为有效
    # pcd = pcd.select_by_index(np.where(valid_mask)[0])

    return pcd

# import sys
# import termios
# import tty

# def read_key() -> str:
#     """
#     非阻塞读取单个字符（Linux / macOS 终端）。
#     Windows 下可改用 msvcrt.getch。
#     """
#     fd = sys.stdin.fileno()
#     old_settings = termios.tcgetattr(fd)
#     try:
#         tty.setraw(sys.stdin.fileno())
#         ch = sys.stdin.read(1)
#     finally:
#         termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#     return ch
# # =================================================


# def capture_and_show(camera):
#     """
#     获取单帧并显示点云
#     """
#     color_frame, depth_frame = camera.get_frame()
#     if color_frame is None or depth_frame is None:
#         print("Failed to get frame, abort.")
#         return

#     intrinsics = camera.get_camera_intrinsic()
#     color_intrinsics = intrinsics['color_intrinsics']
#     color_intrinsics_dict = {
#         'fx': color_intrinsics.fx,
#         'fy': color_intrinsics.fy,
#         'cx': color_intrinsics.cx,
#         'cy': color_intrinsics.cy,
#         'width': color_intrinsics.width,
#         'height': color_intrinsics.height
#     }

#     pcd = rgbd2pcd_full_points(color_frame, depth_frame, color_intrinsics_dict)
#     print("点云加载成功，点数：", len(pcd.points))
#     o3d.visualization.draw_geometries([pcd])



# if __name__ == "__main__":
#     print("按 m 键捕获并显示点云，按 q 键退出。")
#     with OrbbecCamera(ip="192.168.1.10") as camera:
#         while True:
#             key = read_key()
#             if key.lower() == 'm':
#                 capture_and_show(camera)
#             elif key.lower() == 'q':
#                 print("退出。")
#                 break




# 使用示例
if __name__ == "__main__":
    vis_type = "1pointcloud"
    with OrbbecCamera(ip="192.168.1.124") as camera:
        intrinsics = camera.get_camera_intrinsic()
        print("Full intrinsics:", intrinsics)
        
        color_distcodff = camera.get_distCoeffs()
        print("Color distortion coefficients:", color_distcodff)
        
        color_intrinsics_mat = camera.get_intrinsics_mat()
        print("Color intrinsics matrix:", color_intrinsics_mat)

        while True:
            if vis_type == "pointcloud":
                color_frame, depth_frame = camera.get_frame()
                
            else:
                color_frame, depth_frame = camera.get_frame()
                if depth_frame is not None and color_frame is not None:
                    # 显示深度图像
                    depth_image = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
                    cv2.imshow("Depth Viewer", depth_image)
                    x, y, cw, ch = [350,80,550,390]
                    def safe_crop(img):
                        H, W = img.shape[:2]
                        x0 = max(0, int(x))
                        y0 = max(0, int(y))
                        x1 = min(W, x0 + int(cw))
                        y1 = min(H, y0 + int(ch))
                        if x1 <= x0 or y1 <= y0:
                            return img  # 无效裁剪则返回原图
                        return img[y0:y1, x0:x1]

                    color_frame = safe_crop(color_frame)

                    
                    # 显示彩色图像
                    cv2.imshow("Color Viewer", color_frame)
                    
                    # 检查退出键
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == ESC_KEY:
                        break
                    
                    print(f"Color frame: {color_frame.shape}, Depth frame: {depth_frame.shape}")
                else:
                    print("Failed to get frame, retrying...")
                    time.sleep(0.1)
