import numpy as np
import cv2
import time
from pyorbbecsdk import *
from gym_hil.envs.utils_orbbec import frame_to_rgb_image  
import open3d as o3d

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
    def __init__(self, ip="192.168.5.124", port=8090):
        self.ctx = Context()
        self.device = self.ctx.create_net_device(ip, port)
        if not self.device:
            raise ConnectionError("Failed to create net device")
        
        self.pipeline = Pipeline(self.device)
        self.config = Config()
        self.temporal_filter = TemporalFilter(alpha=0.5)
        self.initialize()
        self.start()
        time.sleep(1)

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
        
        # 创建对齐过滤器（深度对齐到彩色）
        self.align_filter = AlignFilter(OBStreamType.COLOR_STREAM)

    def start(self):
        self.pipeline.start(self.config)
        self.pipeline.enable_frame_sync()  # 启用帧同步
        self.running = True
    
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

    def get_frame(self, timeout=100):
        start_time =time.time()

        while True:
            frames = self.pipeline.wait_for_frames(timeout)
            if not frames:
                continue
            current_time = time.time()
            if current_time - start_time>1:
                print("get frame timeout")
                return None,None
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if depth_frame is None or color_frame is None:
                continue
            
            depth_format = depth_frame.get_format()
            if depth_format != OBFormat.Y16:
                print("depth format is not Y16")
                return None, None
            
            aligned_frames = self.align_filter.process(frames)
            
            aligned_frames = aligned_frames.as_frame_set()
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where((depth_data > 10) & (depth_data < 10000), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)
            
            # depth_data = self.temporal_filter.process(depth_data)
            
            color_image = frame_to_rgb_image(color_frame)
            return color_image, depth_data
            

    def stop(self):
        self.running = False
        self.pipeline.stop()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
def rgbd2pcd(rgb, depth, depth_intrinsics_dict):
    """
    将RGB和深度图像转换为点云
    参数:
        rgb: np.ndarray, HWC, 8位uint8, RGB彩色
        depth: np.ndarray, HW, 16位uint16, 单位为毫米(mm)或者float, 单位为米
        depth_intrinsics_dict: dict, 包含fx, fy, cx, cy, width, height
    返回:
        o3d.geometry.PointCloud 对象
    """
    # 如果深度是uint16（mm），转为米
    if depth.dtype == np.uint16:
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32) / 1000.0)
    elif np.issubdtype(depth.dtype, np.floating):
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    else:
        raise ValueError("Unsupported depth dtype")

    color_o3d = o3d.geometry.Image(rgb)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        convert_rgb_to_intensity=False,
        depth_scale=1.0,  # 因为已转到米
        depth_trunc=3.0  # 可设置最大深度
    )

    # 构造内参 - 关键修改：对齐后使用彩色相机内参
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        depth_intrinsics_dict['width'],
        depth_intrinsics_dict['height'],
        depth_intrinsics_dict['fx'],
        depth_intrinsics_dict['fy'],
        depth_intrinsics_dict['cx'],
        depth_intrinsics_dict['cy']
    )

    # 生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic
    )

    return pcd

# 使用示例
if __name__ == "__main__":
    vis_type = "1pointcloud"
    with OrbbecCamera(ip="192.168.5.124") as camera:
        intrinsics = camera.get_camera_intrinsic()
        print("Full intrinsics:", intrinsics)
        
        color_distcodff = camera.get_distCoeffs()
        print("Color distortion coefficients:", color_distcodff)
        
        color_intrinsics_mat = camera.get_intrinsics_mat()
        print("Color intrinsics matrix:", color_intrinsics_mat)

        while True:
            if vis_type == "pointcloud":
                color_frame, depth_frame = camera.get_frame()
                
                if color_frame is None or depth_frame is None:
                    print("Failed to get frame, retrying...")
                    continue

                # 关键修改：对齐后使用彩色相机内参进行点云转换
                intrinsics = camera.get_camera_intrinsic()
                color_intrinsics = intrinsics['color_intrinsics']
                color_intrinsics_dict = {
                    'fx': color_intrinsics.fx,
                    'fy': color_intrinsics.fy,
                    'cx': color_intrinsics.cx,
                    'cy': color_intrinsics.cy,
                    'width': color_intrinsics.width,
                    'height': color_intrinsics.height
                }

                # 使用对齐后的RGB和深度图生成点云
                pcd_from_rgbd = rgbd2pcd(color_frame, depth_frame, color_intrinsics_dict)

                # 可视化点云
                o3d.visualization.draw_geometries([pcd_from_rgbd])
            else:
                color_frame, depth_frame = camera.get_frame()
                if depth_frame is not None and color_frame is not None:
                    # 显示深度图像
                    depth_image = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
                    cv2.imshow("Depth Viewer", depth_image)
                    
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
