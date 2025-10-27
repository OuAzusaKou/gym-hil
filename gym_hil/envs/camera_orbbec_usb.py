import cv2
import numpy as np
import time
from queue import Queue
import threading
import logging

try:
    from pyorbbecsdk import *
    from gym_hil.envs.utils_orbbec import frame_to_bgr_image, frame_to_rgb_image
    ORBBEC_AVAILABLE = True
except ImportError:
    ORBBEC_AVAILABLE = False
    logging.warning("pyorbbecsdk not available. Camera class will not be functional.")
    # Create dummy functions for type hints
    def frame_to_bgr_image(frame):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    def frame_to_rgb_image(frame):
        return np.zeros((480, 640, 3), dtype=np.uint8)

import open3d as o3d
import os
MAX_QUEUE_SIZE = 1
ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm
class Camera():
    def __init__(self, sn="CP7X54P00084",camera_width=1920, camera_height=1080, camera_fps=30, enable_sync=True,
                 align_filter = None, temporal_alpha = 0.5, 
                 sensor_type = None):
        if not ORBBEC_AVAILABLE:
            raise ImportError("pyorbbecsdk not available. Please install pyorbbecsdk to use Camera.")
        
        # Set defaults using OBStreamType and OBSensorType
        if align_filter is None:
            align_filter = OBStreamType.COLOR_STREAM
        if sensor_type is None:
            sensor_type = {OBSensorType.COLOR_SENSOR:[640, 480, OBFormat.RGB, 30],
                           OBSensorType.DEPTH_SENSOR:[640, 480, OBFormat.Y16, 30]}
        
        ctx = Context()
        self.device =None
        device_list = ctx.query_devices()
        time.sleep(0.5)
        for i in range(device_list.get_count()):
            dev = device_list.get_device_by_index(i)
            sn_ = dev.get_device_info().get_serial_number()
            print(f"found index ={i},serial={sn_}")
            if(sn_ == sn):
                self.device = device_list.get_device_by_serial_number(sn)  
   #     if(self.device == None):
   #         print(f"current {sn} device is not founded")
   #         return
        print(f"input sn is:{sn}")
        self.pipeline = Pipeline(self.device)
        self.config = Config()
        self.align_filter = AlignFilter(align_to_stream = align_filter)
        self.temporal_filter = self.TemporalFilter(alpha = temporal_alpha)
        self.point_cloud_filter = PointCloudFilter()
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.enable_sync = enable_sync
        self.dev_name = "orbbec"
  #      self.device = self.pipeline.get_device()
#        print(f"self device is :{self.device}")
        self.sensor_list = self.device.get_sensor_list()
#        print(f"self.sensor_list is:{self.sensor_list}")       
        self.video_sensor_types = [
            OBSensorType.DEPTH_SENSOR,
            OBSensorType.COLOR_SENSOR
        ]
        self.latest_frames = {"depth": None, "color": None}
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)
        self.running = False
        self.processing_thread = None
        self.dist_coeffs = None
        self.color_intrinsics_mat = None
        self.stop_rendering = False
        self.sensor = sensor_type
        self.first_frame = True
        self.init_profile = {}
        self.opened = False
        self.has_color_sensor = False
        self.init_profile_list(sensor = self.sensor)
        self.start()
        self.start_processing()
    
    def start_processing(self):
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
    
    def _process_frames(self):
        start_time =time.time()
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(100)
                if not frames:
                    continue
                current_time = time.time()
                if current_time - start_time<1:
                    continue
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    continue
                # depth_profile = depth_frame.get_stream_profile()
                # depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
                # print("depth_intrinsics is:",depth_intrinsics)
                color_frame = frames.get_color_frame()

                depth_format = depth_frame.get_format()
                if depth_format != OBFormat.Y16:
                    print("depth format is not Y16")
                    continue
                
                frames = self.align_filter.process(frames)
                if not frames:
                    continue
                frames  = frames.as_frame_set()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
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
                # print("depth_data is: ",depth_data)
                # depth_data_raw = depth_data.copy()
                depth_data = self.temporal_filter.process(depth_data)                
                
                # Process color frame
                color_image = frame_to_bgr_image(color_frame)
                
                with self.frame_ready:
                    self.latest_frames["depth"] = depth_data
                    self.latest_frames["color"] = color_image
                    self.frame_ready.notify_all()
            
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue
    
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
    
    def init_profile_list(self, **sensor_kwargs):
        for sensor in range(len(self.sensor_list)):
            sensor_type = self.sensor_list[sensor].get_type()
            try:
                if sensor_type in sensor_kwargs['sensor']:
                    sensor_profile_list = self.pipeline.get_stream_profile_list(sensor_type)
                    if sensor_profile_list is None:
                        print(f"No proper {sensor_type} profile, cannot generate point cloud")
                        return
                    video_config = sensor_kwargs['sensor'][sensor_type]
                    sensor_profile = sensor_profile_list.get_video_stream_profile(video_config[0], video_config[1], video_config[2], video_config[3])
                    self.init_profile[sensor_type] = sensor_profile
                    self.config.enable_stream(sensor_profile)
                    print(f"Enabled custom stream: {sensor_type}: {sensor_profile}")
                if sensor_type in self.video_sensor_types and sensor_type not in sensor_kwargs['sensor']:
                    sensor_profile_list = self.pipeline.get_stream_profile_list(sensor_type)
                    if sensor_profile_list is None:
                        print(f"No proper {sensor_type} profile, cannot generate point cloud")
                        return
                    sensor_profile = sensor_profile_list.get_default_video_stream_profile()
                    self.init_profile[sensor_type] = sensor_profile
                    self.config.enable_stream(sensor_profile)
                    print(f"Enabled default stream: {sensor_type}: {sensor_profile}")
                if sensor_type == OBSensorType.COLOR_SENSOR:
                    self.has_color_sensor = True
            except Exception as e:
                print(f"Failed to enable sensor type: {sensor_type}: {str(e)}")
                continue
    
    def start(self):
        try:
            if self.enable_sync:
                self.pipeline.enable_frame_sync()
                print("Enable Frame Sync")
            self.pipeline.start(self.config)
            self.opened = True
            print("Camera started successfully.")
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
    
    @property
    def get_camera_intrinsics(self):
        intrinsics = {}
        if self.init_profile.keys():
            # Get depth intrinsics
            depth_intrinsics = self.init_profile[OBSensorType.DEPTH_SENSOR].get_intrinsic()
            intrinsics['depth_intrinsics'] = depth_intrinsics
            
            # Get depth distortion parameter
            depth_distortion = self.init_profile[OBSensorType.DEPTH_SENSOR].get_distortion()
            intrinsics['depth_distortion'] = depth_distortion
            
            # Get color internal parameters
            color_intrinsics = self.init_profile[OBSensorType.COLOR_SENSOR].get_intrinsic()
            intrinsics['color_intrinsics'] = color_intrinsics
            
            # Get color distortion parameter
            color_distortion = self.init_profile[OBSensorType.COLOR_SENSOR].get_distortion()
            intrinsics['color_distortion'] = color_distortion
        return intrinsics
    
    def get_distCoeffs(self):
        intrinsics = self.get_camera_intrinsics
        if intrinsics is not None:
            temp = intrinsics["color_distortion"]
            color_distortion = [temp.k1, temp.k2, temp.k3, temp.k4, temp.k5, temp.k6, temp.p1, temp.p2]
        self.dist_coeffs = np.asarray(color_distortion)
        return self.dist_coeffs
    
    def get_intrinsics_mat(self):
        intrinsics = self.get_camera_intrinsics
        if intrinsics is not None:
            fx = intrinsics["color_intrinsics"].fx
            fy = intrinsics["color_intrinsics"].fy
            cx = intrinsics["color_intrinsics"].cx
            cy = intrinsics["color_intrinsics"].cy
            self.color_intrinsics_mat = np.array([[fx, 0, cx], 
                                                [0, fy, cy], 
                                                [0, 0, 1]])
        return self.color_intrinsics_mat
    
    def get_frame(self, timeout=None):
        """返回最新的BGR图和深度图（单位：毫米）"""
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
        try:
            if self.opened:
                self.pipeline.stop()
                cv2.destroyAllWindows()
                self.opened = False
                print("Camera stopped successfully.")
        except Exception as e:
            print(f"Error stopping camera: {str(e)}")
    
    def __enter__(self):
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


if __name__ == "__main__":
    vis_type = "pointcloud1"

    camera = Camera()
    current_time = time.time()
    last_time = time.time()
    cnt = 0
    frame_rate = 0
    print(camera.get_camera_intrinsics)  # 获取相机内外参
    print(camera.get_distCoeffs())  # 获取相机畸变系数
    print(camera.get_intrinsics_mat())  # 获取相机内参矩阵

    while True:
        if vis_type == "pointcloud":
            color_frame, depth_frame = camera.get_frame()

            # 关键修改：对齐后使用彩色相机内参进行点云转换
            intrinsics = camera.get_camera_intrinsics
            color_intrinsics = intrinsics['color_intrinsics']
            color_intrinsics_dict = {}
            color_intrinsics_dict['fx'] = color_intrinsics.fx
            color_intrinsics_dict['fy'] = color_intrinsics.fy
            color_intrinsics_dict['cx'] = color_intrinsics.cx
            color_intrinsics_dict['cy'] = color_intrinsics.cy
            color_intrinsics_dict['width'] = color_intrinsics.width
            color_intrinsics_dict['height'] = color_intrinsics.height

            # 使用对齐后的RGB和深度图生成点云
            pcd_from_rgbd = rgbd2pcd(color_frame, depth_frame, color_intrinsics_dict)

            # 可视化点云
            o3d.visualization.draw_geometries([pcd_from_rgbd])

        else:
            try:
                color_frame, depth_frame = camera.get_frame()
                current_time = time.time()
                cnt += 1

                # 计算帧率
                elapsed_time = current_time - last_time
                if elapsed_time >= 1.0:
                    frame_rate = cnt / elapsed_time
                    # print(f"Frame Rate: {frame_rate:.2f} FPS")
                    last_time = current_time
                    cnt = 0

                # 显示彩色图像
                bgr_img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("Color Viewer", bgr_img)
                # cv2.imwrite("depth_image.png", depth_frame)

                # 显示深度图像
                depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                depth_display = depth_display.astype(np.uint8)
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow("Depth Viewer", depth_display)

                key = cv2.waitKey(1)
                if key == ord('q') or key == ESC_KEY:
                    break

            except TimeoutError:
                print("Timeout waiting for frames")
                continue

    cv2.destroyAllWindows()

    camera.stop()

