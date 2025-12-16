import numpy as np
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
import cv2
ESC_KEY = 27
class RSCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name, serial_number='323622271380', dim=(640, 480), fps=15, depth=False, exposure=40000):
        self.name = name
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)
        self.s = self.profile.get_device().query_sensors()[0]
        self.s.set_option(rs.option.exposure, exposure)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            image = np.asarray(color_frame.get_data())
            if self.depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((image, depth), axis=-1)
            else:
                return True, image
        else:
            return False, None

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()

import queue
import threading
import time

class VideoCapture:
    def __init__(self, cap, name=None):
        if name is None:
            name = cap.name
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = False
        self.enable = True
        self.t.start()

    def _reader(self):
        while self.enable:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        # print(self.name, self.q.qsize())
        return self.q.get(timeout=1)
    
    def get_frame(self):
        return self.read(),None

    def close(self):
        self.enable = False
        self.t.join()
        self.cap.close()

class Camera():
    def __init__(self):
        self.video_capture = VideoCapture(RSCapture(name="hand")
                )
    
    def get_frame(self):
        return self.video_capture.get_frame()
    
    def close(self):
        self.video_capture.close()


if __name__ == "__main__":
    camera = VideoCapture(RSCapture(name="hand")
                )
    while True:


        # if cam_name == "side_classifier":
        #     self.cap["side_classifier"] = self.cap["side_policy"]
        # else:

            # cap.read()
        try:
            color_frame,_ = camera.get_frame()
            current_time = time.time()
            # cnt += 1

            # 计算帧率
            # elapsed_time = current_time - last_time
            # if elapsed_time >= 1.0:
            #     frame_rate = cnt / elapsed_time
            #     # print(f"Frame Rate: {frame_rate:.2f} FPS")
            #     last_time = current_time
            #     cnt = 0

            # 显示彩色图像
            bgr_img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            
            x, y, cw, ch = [100,200,300,200]
            def safe_crop(img):
                H, W = img.shape[:2]
                print("H, W", H, W)
                x0 = max(0, int(x))
                y0 = max(0, int(y))
                x1 = min(W, x0 + int(cw))
                y1 = min(H, y0 + int(ch))
                print("x0, y0, x1, y1", x0, y0, x1, y1)
                if x1 <= x0 or y1 <= y0:
                    return img  # 无效裁剪则返回原图
                return img[y0:y1, x0:x1]

            color_frame = safe_crop(color_frame)
            cv2.imshow("Color Viewer", color_frame)
            # cv2.imwrite("depth_image.png", depth_frame)

            # 显示深度图像
            # # depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            # depth_display = depth_display.astype(np.uint8)
            # depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            # cv2.imshow("Depth Viewer", depth_display)

            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break

        except TimeoutError:
            print("Timeout waiting for frames")
            continue

    cv2.destroyAllWindows()

    camera.stop()