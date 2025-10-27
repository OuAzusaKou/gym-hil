import cv2
import numpy as np
import time
from queue import Queue
from pyorbbecsdk import *
from utils_orbbec import frame_to_bgr_image, frame_to_rgb_image
import open3d as o3d
import os
MAX_QUEUE_SIZE = 1
ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

class Camera():
    def __init__(self, camera_width=1920, camera_height=1080, camera_fps=30, enable_sync=True, 
                 align_filter = OBStreamType.COLOR_STREAM, temporal_alpha = 0.1, 
                 sensor_type = {OBSensorType.COLOR_SENSOR:[640, 480, OBFormat.RGB, 30], 
                                OBSensorType.DEPTH_SENSOR:[640, 480, OBFormat.Y16, 30]}):
        self.pipeline = Pipeline()
        self.config = Config()
        self.align_filter = AlignFilter(align_to_stream = align_filter)
        self.temporal_filter = self.TemporalFilter(alpha = temporal_alpha)
        self.point_cloud_filter = PointCloudFilter()
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.enable_sync = enable_sync
        self.dev_name = "orbbec"
        self.device = self.pipeline.get_device()
        self.sensor_list = self.device.get_sensor_list()
        self.video_sensor_types = [
            OBSensorType.DEPTH_SENSOR,
            # OBSensorType.LEFT_IR_SENSOR,
            # OBSensorType.RIGHT_IR_SENSOR,
            # OBSensorType.IR_SENSOR,
            OBSensorType.COLOR_SENSOR
        ]
        self.cached_frames = {
            'color': None,
            'depth': None,
            'left_ir': None,
            'right_ir': None,
            'ir': None
        }
        self.processed_frames = dict()
        self.frames_queue = Queue()
        self.dist_coeffs = None
        self.color_intrinsics_mat = None  # 初始化为 None
        self.stop_rendering = False
        self.sensor = sensor_type
        self.first_frame = True
        self.init_profile = {}
        self.opened = False
        self.has_color_sensor = False
        self.init_profile_list(sensor = self.sensor)
        self.start()
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
                print(sensor_type)
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
        # profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        # #Get color_profile
        # # color_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        # color_profile = profile_list.get_default_video_stream_profile()
        # profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        # #Get depth_profile
        # depth_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 6)
        # # depth_profile = profile_list.get_default_video_stream_profile()
        #Get external parameters
        intrinsics = {}
        if self.init_profile.keys():
            extrinsic = self.init_profile[OBSensorType.DEPTH_SENSOR].get_extrinsic_to(self.init_profile[OBSensorType.DEPTH_SENSOR])
            intrinsics['extrinsic'] = extrinsic
            print("extrinsic  {}".format(extrinsic))
            #Get depth inernal parameters
            depth_intrinsics = self.init_profile[OBSensorType.DEPTH_SENSOR].get_intrinsic()
            intrinsics['depth_intrinsics'] = depth_intrinsics
            print("depth_intrinsics  {}".format(depth_intrinsics))
            #Get depth distortion parameter
            depth_distortion = self.init_profile[OBSensorType.DEPTH_SENSOR].get_distortion()
            intrinsics['depth_distortion'] = depth_distortion
            print("depth_distortion  {}".format(depth_distortion))
            #Get color internala parameters
            color_intrinsics = self.init_profile[OBSensorType.COLOR_SENSOR].get_intrinsic()
            intrinsics['color_intrinsics'] = color_intrinsics
            print("color_intrinsics  {}".format(color_intrinsics))
            #Get color distortion parameter
            color_distortion = self.init_profile[OBSensorType.COLOR_SENSOR].get_distortion()
            intrinsics['color_distortion'] = color_distortion
            print("color_distortion  {}".format(color_distortion))
        return intrinsics
    def get_distCoeffs(self):
        intrinsics = self.get_camera_intrinsics
        if intrinsics is not None:
            temp = intrinsics["color_distortion"]
            # color_distortion = [temp.k1, temp.k2, temp.k3, temp.k4, temp.k5, temp.k6, temp.p1, temp.p2]
            color_distortion = {"k1": temp.k1, "k2": temp.k2, "k3": temp.k3, "k4": temp.k4, "k5": temp.k5, "k6": temp.k6, "p1": temp.p1, "p2": temp.p2}
        # self.dist_coeffs = np.asarray(color_distortion)
        self.dist_coeffs = color_distortion
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
    def capture_frame(self):
        if not self.opened or self.stop_rendering:
            return
        while True:
            try:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    print("faild to capture frame, retry...")
                    continue
                if self.first_frame:
                    print("first frame captured")
                    self.first_frame = False
                    continue
                break
            except KeyboardInterrupt:
                return
        return frames
    def get_rgb_image(self, frames, img_type = "RGB"):
        try:
            # frames: FrameSet = self.pipeline.wait_for_frames(100)
            if img_type not in ["RGB", "BGR"]:
                raise ValueError("img_type must be 'RGB' or 'BGR'")
            if frames is None:
                return
            color_frame = frames.get_color_frame()
            if color_frame is None:
                return
            # covert to RGB formatr
            # color_image = frame_to_bgr_image(color_frame)
            if img_type == "RGB":
                color_image = frame_to_rgb_image(color_frame)
            elif img_type == "BGR":
                color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                return
            self.cached_frames['color'] = color_image
            return color_image
            # cv2.imshow("Color Viewer", color_image)
            # key = cv2.waitKey(1)
            # if key == ord('q') or key == ESC_KEY:
            #     return
        except KeyboardInterrupt:
            return
    def get_depth_image(self, frames, scale = 1.0, temporal_filt = True):
        try:
            # frames: FrameSet = self.pipeline.wait_for_frames(100)
            if frames is None:
                return
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                return
            depth_format = depth_frame.get_format()
            if depth_format != OBFormat.Y16:
                print("depth format is not Y16")
                return
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            # scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))
            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)
            depth_data_raw = depth_data.copy()
            # Apply temporal filtering
            if temporal_filt:
                depth_data = self.temporal_filter.process(depth_data)
            # center_y = int(height / 2)
            # center_x = int(width / 2)
            # center_distance = depth_data[center_y, center_x]

            # current_time = time.time()
            # if current_time - last_print_time >= PRINT_INTERVAL:
            #     print("center distance: ", center_distance)
            #     last_print_time = current_time

            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
            self.cached_frames['depth'] = depth_image
            return depth_image, depth_data_raw
            # cv2.imshow("Depth Viewer", depth_image)
            # key = cv2.waitKey(1)
            # if key == ord('q') or key == ESC_KEY:
            #     return
        except KeyboardInterrupt:
            return
    def get_point_cloud(self, frames, color_frame = None, depth_frame = None):
        if frames is None:
            return
        if depth_frame is None:
            depth_frame = self.cached_frames['depth']
            if depth_frame is None:
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    return
        if color_frame is None:
            color_frame = self.cached_frames['color']
            if color_frame is None:
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    return
        
        frame = self.align_filter.process(frames)
        #scale = depth_frame.get_depth_scale()
        #point_cloud_filter.set_position_data_scaled(scale)

        point_format = OBFormat.RGB_POINT if self.has_color_sensor and color_frame is not None else OBFormat.POINT
        self.point_cloud_filter.set_create_point_format(point_format)

        point_cloud_frame = self.point_cloud_filter.process(frame)
        if point_cloud_frame is None:
            return
        #save point cloud
        points = self.point_cloud_filter.calculate(point_cloud_frame)
        # points_coords = np.array([p[:3] for p in points]) 
        # points_color = np.array([p[3:6]/255 for p in points])
        # pcd_with_color = o3d.geometry.PointCloud()
        # pcd_with_color.points = o3d.utility.Vector3dVector(points_coords)
        # pcd_with_color.colors = o3d.utility.Vector3dVector(points_color)
        # output_filepath_with_color = "output_point_cloud_with_color.ply"
        # o3d.io.write_point_cloud(output_filepath_with_color, pcd_with_color)
        # pcd_loaded_with_color = o3d.io.read_point_cloud(output_filepath_with_color)
        # o3d.visualization.draw_geometries([pcd_loaded_with_color], window_name="带颜色点云")
        # save_point_cloud_to_ply(os.path.join("point_cloud.ply"), point_cloud_frame)
        # return point_cloud_frame
        return points
    def stop(self):
        try:
            if self.opened:
                self.zed.close()
                cv2.destroyAllWindows()
                self.opened = False
                print("Camera stopped successfully.")
        except Exception as e:
            print(f"Error stopping camera: {str(e)}")

    def display_images(self):
        while self.opened:
            image_np, depth_np = self.get_image()
            if image_np is None or depth_np is None:
                continue

            cv2.imshow("RGB Image", image_np)

            depth_display = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype(np.uint8)

            cv2.imshow("Depth Image", depth_display)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    def on_new_frame_callback(self, frame: FrameSet):
        """Callback function to handle new frames"""
        if frame is None:
            return
        if self.frames_queue.qsize() >= MAX_QUEUE_SIZE:
            self.frames_queue.get()
        self.frames_queue.put(frame)


    def process_color(self, frame):
        """Process color frame to BGR image"""
        if not frame:
            return None
        color_frame = frame.get_color_frame()
        color_frame = color_frame if color_frame else self.cached_frames['color']
        if not color_frame:
            return None
        try:
            self.cached_frames['color'] = color_frame
            return frame_to_bgr_image(color_frame)
        except ValueError:
            print("Error processing color frame")
            return None


    def process_depth(self, frame):
        """Process depth frame to colorized depth image"""
        if not frame:
            return None
        # self.pipeline.wait_for_frames(100)
        depth_frame = frame.get_depth_frame()
        depth_frame = depth_frame if depth_frame else self.cached_frames['depth']
        if not depth_frame:
            return None
        try:
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape(depth_frame.get_height(), depth_frame.get_width())
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.cached_frames['depth'] = depth_frame
            return cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        except ValueError:
            print("Error processing depth frame")
            return None


    def process_ir(self, frame, frame_type):
        if frame is None:
            return None
        ir_frame = frame.get_frame(frame_type)
        frame_name = 'ir' if frame_type == OBFrameType.IR_FRAME else 'left_ir' if frame_type == OBFrameType.LEFT_IR_FRAME else 'right_ir'
        ir_frame = ir_frame if ir_frame else self.cached_frames[frame_name]
        if not ir_frame:
            return None
        ir_frame = ir_frame.as_video_frame()
        self.cached_frames[frame_name] = ir_frame
        ir_data = np.asanyarray(ir_frame.get_data())
        width = ir_frame.get_width()
        height = ir_frame.get_height()
        ir_format = ir_frame.get_format()

        if ir_format == OBFormat.Y8:
            ir_data = np.resize(ir_data, (height, width, 1))
            data_type = np.uint8
            image_dtype = cv2.CV_8UC1
            max_data = 255
        elif ir_format == OBFormat.MJPG:
            ir_data = cv2.imdecode(ir_data, cv2.IMREAD_UNCHANGED)
            data_type = np.uint8
            image_dtype = cv2.CV_8UC1
            max_data = 255
            if ir_data is None:
                print("decode mjpeg failed")
                return None
            ir_data = np.resize(ir_data, (height, width, 1))
        else:
            ir_data = np.frombuffer(ir_data, dtype=np.uint16)
            data_type = np.uint16
            image_dtype = cv2.CV_16UC1
            max_data = 255
            ir_data = np.resize(ir_data, (height, width, 1))

        cv2.normalize(ir_data, ir_data, 0, max_data, cv2.NORM_MINMAX, dtype=image_dtype)
        ir_data = ir_data.astype(data_type)
        return cv2.cvtColor(ir_data, cv2.COLOR_GRAY2RGB)



    def create_display(self, processed_frames, width=1280, height=720):
        """Create display window with all processed frames
        Layout:
        2x2 grid when both left and right IR are present:
        [Color] [Depth]
        [L-IR] [R-IR]

        2x2 grid with single IR:
        [Color] [Depth]
        [  IR  ][     ]
        """
        display = np.zeros((height, width, 3), dtype=np.uint8)
        h, w = height // 2, width // 2

        # Helper function for safe image resizing
        def safe_resize(img, target_size):
            if img is None:
                return None
            try:
                return cv2.resize(img, target_size)
            except:
                return None

        # Process frames with consistent error handling
        def place_frame(img, x1, y1, x2, y2):
            if img is not None:
                try:
                    h_section = y2 - y1
                    w_section = x2 - x1
                    resized = safe_resize(img, (w_section, h_section))
                    if resized is not None:
                        display[y1:y2, x1:x2] = resized
                except:
                    pass

        # Always show color and depth in top row if available
        place_frame(processed_frames.get('color'), 0, 0, w, h)
        place_frame(processed_frames.get('depth'), w, 0, width, h)

        # Handle IR display in bottom row
        has_left_ir = processed_frames.get('left_ir') is not None
        has_right_ir = processed_frames.get('right_ir') is not None
        has_single_ir = processed_frames.get('ir') is not None

        if has_left_ir and has_right_ir:
            # Show stereo IR in bottom row
            place_frame(processed_frames['left_ir'], 0, h, w, height)
            place_frame(processed_frames['right_ir'], w, h, width, height)
        elif has_single_ir:
            # Show single IR in bottom-left quadrant
            place_frame(processed_frames['ir'], 0, h, w, height)

        # Add labels to identify each stream
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255)
        font_thickness = 2

        # Helper function for adding labels
        def add_label(text, x, y):
            cv2.putText(display, text, (x + 10, y + 30), font, font_scale,
                    font_color, font_thickness)

        # Add labels for each quadrant
        add_label("Color", 0, 0)
        add_label("Depth", w, 0)

        if has_left_ir and has_right_ir:
            add_label("Left IR", 0, h)
            add_label("Right IR", w, h)
        elif has_single_ir:
            add_label("IR", 0, h)

        return display

    def rendering_frames(self):
        """Main rendering loop for processing and displaying frames"""
        cnt = 0
        # Create and configure display window
        cv2.namedWindow("Orbbec Camera Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Orbbec Camera Viewer", 1280, 720)

        while not self.stop_rendering:
            if self.frames_queue.empty():
                continue

            frame_set = self.frames_queue.get()
            if frame_set is None:
                continue
            frame_set = self.align_filter.process(frame_set)
            frame_set  = frame_set.as_frame_set()
            # Process all available frames
            self.processed_frames = {
                'color': self.process_color(frame_set),
                'depth': self.process_depth(frame_set),
            }

            # Process IR frames with better error handling
            try:
                left_ir = self.process_ir(frame_set, OBFrameType.LEFT_IR_FRAME)
                right_ir = self.process_ir(frame_set, OBFrameType.RIGHT_IR_FRAME)
                if left_ir is not None and right_ir is not None:
                    self.processed_frames['left_ir'] = left_ir
                    self.processed_frames['right_ir'] = right_ir
                else:
                    # Try single IR if stereo IR is not available
                    ir = self.process_ir(frame_set, OBFrameType.IR_FRAME)
                    if ir is not None:
                        self.processed_frames['ir'] = ir
            except:
                # Fallback to single IR in case of any error
                try:
                    ir = self.process_ir(frame_set, OBFrameType.IR_FRAME)
                    if ir is not None:
                        self.processed_frames['ir'] = ir
                except:
                    pass
            print(self.processed_frames.keys())
            # Create and display the combined view
            if self.processed_frames['color'] is not None and self.processed_frames['depth'] is not None:
                color_img = self.processed_frames['color']
                depth_img = self.processed_frames['depth']

                # Display raw RGB and Depth images using OpenCV
                combined_display = np.zeros((480, 1280, 3), dtype=np.uint8)
                combined_display[:480, :640] = color_img  # Top-left: RGB
                combined_display[:480, 640:] = depth_img  # Top-right: Depth (colorized)
                cv2.imshow("Orbbec Camera Viewer", combined_display)
                # Optionally, create RGBD image for Open3D visualization (not displayed in OpenCV)
                color_o3d = o3d.geometry.Image(self.processed_frames['color'])
                depth_o3d = o3d.geometry.Image(self.processed_frames['depth'])
                print("Open3D Color Image:", color_o3d)
                print("Open3D Depth Image:", depth_o3d)
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d, depth_scale=1000, convert_rgb_to_intensity=False)

                print(rgbd_image)
                fx = 459.528595
                fy = 459.652069
                cx = 321.709106
                cy = 246.846436
                intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)
                print(intrinsics)
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
                o3d.visualization.draw_geometries([pcd])
            display = self.create_display(self.processed_frames)
            cv2.imshow("Orbbec Camera Viewer", display)

            # Check for exit key
            key = cv2.waitKey(1)
            if key in [ord('q'), ESC_KEY]:
                return
    def generate_point_cloud(self, points):
        # 该函数用于从输入的点云数据生成带颜色的点云，并将其保存为PLY格式文件
        points_coords = np.array([p[:3] for p in points]) 
        points_color = np.array([p[3:6]/255 for p in points])
        pcd_with_color = o3d.geometry.PointCloud()
        pcd_with_color.points = o3d.utility.Vector3dVector(points_coords)
        pcd_with_color.colors = o3d.utility.Vector3dVector(points_color)
        output_filepath_with_color = "./output_point_cloud_with_color.ply"
        o3d.io.write_point_cloud(output_filepath_with_color, pcd_with_color)
        pcd_loaded_with_color = o3d.io.read_point_cloud(output_filepath_with_color)
        o3d.visualization.draw_geometries([pcd_loaded_with_color], window_name="带颜色点云")
# def main():
#     """Main function to initialize and run the camera viewer"""
#     # config = Config()
#     # pipeline = Pipeline()

#     try:
#         # Initialize pipeline and config


#         # Get device and sensor information
#         # device = pipeline.get_device()
#         # sensor_list = device.get_sensor_list()

#         # Enable all available video streams
#         profile_list = camera.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
#         #Get color_profile
#         color_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
#         # color_profile = profile_list.get_default_video_stream_profile()
#         profile_list = camera.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
#         depth_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
#         # depth_profile = profile_list.get_default_video_stream_profile()
#         # for sensor in range(len(camera.sensor_list)):
#         #     sensor_type = camera.sensor_list[sensor].get_type()
#         #     print(sensor_type)
#         #     if sensor_type in camera.video_sensor_types and sensor_type not in [OBSensorType.COLOR_SENSOR,
#         #     OBSensorType.DEPTH_SENSOR]:
#         #         try:
#         #             print(f"Enabling sensor type: {sensor_type}")
                    
#         #             camera.config.enable_stream(sensor_type)
#         #         except:
#         #             print(f"Failed to enable sensor type: {sensor_type}")
#         #             continue
#         camera.config.enable_stream(color_profile)
#         camera.config.enable_stream(depth_profile)
#         # Start pipeline with callback
#         camera.pipeline.enable_frame_sync()
#         camera.pipeline.start(camera.config, lambda frames: camera.on_new_frame_callback(frames))

#         # Start rendering frames
#         try:
#             camera.rendering_frames()
#         except KeyboardInterrupt:
#             camera.stop_rendering = True

#     except Exception as e:
#         print(f"Error: {str(e)}")

#         # Cleanup
#     camera.stop_rendering = True
#     camera.pipeline.stop()
#     cv2.destroyAllWindows()
if __name__ == "__main__":
    camera = Camera()
    # camera.start()
    # try:
    #     camera.display_images()
    # finally:
    #     camera.stop()
    # main()

    current_time = time.time()
    last_time = time.time()
    cnt = 0
    print(camera.get_camera_intrinsics) # 获取相机内外参
    print(camera.get_distCoeffs()) # 获取相机畸变系数
    print(camera.get_intrinsics_mat()) # 获取相机内参矩阵
    while current_time - last_time < 100:
        frames = camera.capture_frame() # 获取相机当前数据帧q
        current_time = time.time()
        if frames is None:
            continue
        cnt += 1  
        depth = camera.get_depth_image(frames) # 从数据帧中获取深度图像
        rgb = camera.get_rgb_image(frames) # 从数据帧中获取彩色图像
        if depth is None:
            continue
        cv2.imwrite('calibration_test.png', depth)
        # pcd = camera.get_point_cloud(frames) # 从数据帧中获取点云数据
        # if pcd is not None:
            # camera.generate_point_cloud(pcd) # 生成与可视化点云数据
        #     pass
        # else:
        #     continue
        cv2.imshow("Depth Viewer", depth)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ESC_KEY:
            break
        print(cnt)
        # break
