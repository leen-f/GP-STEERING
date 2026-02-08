import pyzed.sl as sl
import torch
from scipy.spatial.transform import Rotation as R
import csv
import cv2
import math
from datetime import datetime
import pytz
import numpy as np
import threading
import pandas as pd
import mediapipe as mp
import time
import os
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch



jordan = pytz.timezone("Asia/Amman")

NATIONAL_ID = None
BASE_DIR = None

driver_logger = None
crossing_logger = None

import subprocess
import shutil
import os

import os, subprocess

def convert_avi_to_mp4_slow(input_avi, speed=0.20):
    base, _ = os.path.splitext(input_avi)
    output_mp4 = base + "_slowed.mp4"

    if os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 10000:
        return output_mp4

    pts_mult = 1.0 / speed 

    cmd = [
        "ffmpeg", "-y",
        "-i", input_avi,
        "-filter:v", f"setpts={pts_mult}*PTS,scale=1280:-1", 
        "-c:v", "libx264",
        "-preset", "medium", 
        "-crf", "23",  
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_mp4
    ]

    import subprocess
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(output_mp4) and os.path.getsize(output_mp4) > 10000:
        return output_mp4

    return None

def get_national_id():
    global NATIONAL_ID, BASE_DIR
    
    while True:
        national_id = input("Please enter your national ID: ").strip()
        if national_id:
            NATIONAL_ID = national_id
            BASE_DIR = os.path.join(os.getcwd(), NATIONAL_ID)
            
            if not os.path.exists(BASE_DIR):
                os.makedirs(BASE_DIR)
                print(f"Created directory: {BASE_DIR}")
            else:
                print(f"Using existing directory: {BASE_DIR}")
            
            return national_id
        else:
            print("National ID cannot be empty. Please try again.")

def get_filename_with_id(base_name, extension=""):
    global NATIONAL_ID, BASE_DIR
    
    if NATIONAL_ID is None or BASE_DIR is None:
        raise ValueError("National ID not set. Call get_national_id() first.")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if extension:
        filename = f"{NATIONAL_ID}_{base_name}_{timestamp}.{extension}"
    else:
        filename = f"{NATIONAL_ID}_{base_name}_{timestamp}"
    return os.path.join(BASE_DIR, filename)

def init_loggers():
    global driver_logger, crossing_logger
    
    driver_logger = DriverLogger()
    crossing_logger = CrossingLogger()

class DriverLogger:
    def __init__(self):
        self.log_filename = get_filename_with_id("driver_monitoring_log", "txt")
        self.setup_logging()
    
    def setup_logging(self):
        with open(self.log_filename, 'w', encoding='utf-8') as f:
            f.write(f"driver monitoring system log\n")
            f.write(f"started at: {datetime.now(jordan).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
    
    def log_with_timestamp(self, message):
        timestamp = datetime.now(jordan).strftime("%Y-%m-%d %H:%M:%S")
        console_message = f"[{timestamp}] {message}"
        file_message = f"[{timestamp}] {message}"
        
        print(console_message)
        
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(file_message + "\n")

class CrossingLogger:
    def __init__(self):
        self.log_filename = get_filename_with_id("crossing_log", "txt")
        self.setup_logging()

    def setup_logging(self):
        with open(self.log_filename, 'w', encoding='utf-8') as f:
            f.write("crossing detection system log\n")
            f.write(f"started at: {datetime.now(jordan).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

    def log_with_timestamp(self, message):
        timestamp = datetime.now(jordan).strftime("%Y-%m-%d %H:%M:%S")
        console_message = f"[{timestamp}] {message}"
        file_message = f"[{timestamp}] {message}"

        print(console_message)

        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(file_message + "\n")
    
    def log_crossing_event(self, event_type, pedestrian_id, details=""):
        timestamp = datetime.now(jordan).strftime("%Y-%m-%d %H:%M:%S")
        message = f"{event_type} - Pedestrian ID: {pedestrian_id}"
        if details:
            message += f" - {details}"
        
        console_message = f"[{timestamp}] {message}"
        file_message = f"[{timestamp}] {message}"
        
        print(console_message)
        
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(file_message + "\n")



class SeatbeltDetector:
    def __init__(self):
        self.THRESHOLD_SCORE = 0.65
        self.OBJECT_DETECTION_MODEL_PATH = "models/best.pt"
        self.SKIP_FRAMES = 10  
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (255, 0, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        
        self.seatbelt_mark = 0
        self.seatbelt_worn_time = None
        self.evaluation_start_time = None
        self.seatbelt_worn_within_time = False
        
        self.model = None
        self.model_loaded = False
        
        try:
            self._log_with_timestamp("loading YOLOv5 seatbelt model")
            
            if os.path.exists(self.OBJECT_DETECTION_MODEL_PATH):
                self.model = torch.hub.load(
                    "ultralytics/yolov5", 
                    "custom", 
                    path=self.OBJECT_DETECTION_MODEL_PATH, 
                    force_reload=False,  
                    verbose=False,
                    trust_repo=True 
                )
                self._log_with_timestamp(f"YOLOv5 model loaded from {self.OBJECT_DETECTION_MODEL_PATH}")
                self._log_with_timestamp(f"model classes: {self.model.names}")
                self.model_loaded = True
                
            else:
                self._log_with_timestamp(f"YOLOv5 model not found at {self.OBJECT_DETECTION_MODEL_PATH}")
                self._log_with_timestamp("using simulated seatbelt detection")
                
        except Exception as e:
            self._log_with_timestamp(f"error loading seatbelt model: {e}")
            self._log_with_timestamp("using simulated seatbelt detection")
            self.model_loaded = False
        
        self.frame_count = 0
        self.last_seatbelt_status = "Not Detected"
        self.seatbelt_worn_count = 0
        self.seatbelt_not_worn_count = 0
        self.detection_count = 0
        self.detection_active = True
        self.detection_end_time = None 

    def _log_with_timestamp(self, message):
        driver_logger.log_with_timestamp(message)

    def start_evaluation(self):
        self.evaluation_start_time = time.time()
        self.detection_end_time = self.evaluation_start_time + 20  
        self._log_with_timestamp(f"seatbelt evaluation started at {datetime.now().strftime('%H:%M:%S')}")
        self._log_with_timestamp(f"seatbelt detection will be active for 20 seconds (until {datetime.fromtimestamp(self.detection_end_time).strftime('%H:%M:%S')})")

    def calculate_seatbelt_mark(self):
        if self.seatbelt_worn_time is None:
            self._log_with_timestamp("no seatbelt detection --> 0/10")
            return 0
            
        current_time = datetime.now(jordan).strftime("%Y-%m-%d %H:%M:%S")
        time_to_wear = self.seatbelt_worn_time - self.evaluation_start_time
        self._log_with_timestamp(f"seatbelt detected at {current_time}")

        if time_to_wear < 20:
            self.seatbelt_mark = 10
            self.seatbelt_worn_within_time = True
            self._log_with_timestamp("seatbelt detected within 20 seconds: 10/10 marks")
        else:
            self.seatbelt_mark = 0
            self._log_with_timestamp("seatbelt not detected within required time: 0/10 marks")
            
        return self.seatbelt_mark

    def draw_bounding_box(self, img, x1, y1, x2, y2, color):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def draw_text(self, img, x, y, text, color):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def process_frame(self, frame, clean_mode=False):
        self.frame_count += 1
    
        current_time = time.time()
        if self.detection_end_time is not None and current_time > self.detection_end_time:
            if self.detection_active:
                self.detection_active = False
                self._log_with_timestamp("seatbelt detection period ended (20 seconds elapsed)")
            return frame, "Detection Period Ended", 0.0, None 
    
        if not self.detection_active:
            return frame, "Detection Period Ended", 0.0, None  

        if self.frame_count % self.SKIP_FRAMES != 0:
            return frame, self.last_seatbelt_status, 0.0, None  
        seatbelt_status = "Not Detected"
        confidence = 0.0
        bbox = None 

        if self.model_loaded:
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(img_rgb)
                detections = results.pandas().xyxy[0]

                for _, detection in detections.iterrows():
                    if detection['confidence'] > confidence and detection['confidence'] >= self.THRESHOLD_SCORE:
                        confidence = detection['confidence']
                        class_name = detection['name']

                        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), \
                                        int(detection['xmax']), int(detection['ymax'])
                        bbox = (x1, y1, x2, y2)

                        if any(word in class_name.lower() for word in ['seatbelt', 'belt', 'worn', 'buckled']):
                            seatbelt_status = "Worn"
                            draw_color = self.COLOR_GREEN
                            self.seatbelt_worn_count += 1
                        
                            if self.seatbelt_worn_time is None and self.evaluation_start_time is not None:
                                self.seatbelt_worn_time = time.time()
                                self._log_with_timestamp("Seatbelt detected as WORN")
                                self.calculate_seatbelt_mark()

                        elif any(word in class_name.lower() for word in ['no', 'not', 'unbuckled']):
                            seatbelt_status = "Not Worn"
                            draw_color = self.COLOR_RED
                            self.seatbelt_not_worn_count += 1
                            self._log_with_timestamp("Seatbelt detected as NOT WORN")
                        else:
                            seatbelt_status = "Worn"
                            draw_color = self.COLOR_YELLOW
                            self.seatbelt_worn_count += 1

                        if not clean_mode:
                            self.draw_bounding_box(frame, x1, y1, x2, y2, draw_color)
                            status_text = f"{class_name} {confidence:.2f}"
                            self.draw_text(frame, x1, max(y1 - 10, 0), status_text, draw_color)

                        self.detection_count += 1
                        break 
            
                if seatbelt_status == "Not Detected":
                    self.last_seatbelt_status = seatbelt_status

            except Exception as e:
                self._log_with_timestamp(f"Error in seatbelt detection: {e}")
                seatbelt_status = "Error"
        else:
            if self.detection_active and self.frame_count % 60 == 0: 
                if self.last_seatbelt_status == "Worn":
                    seatbelt_status = "Not Worn"
                    self.seatbelt_not_worn_count += 1
                    self._log_with_timestamp("Simulated: Seatbelt NOT WORN")
                else:
                    seatbelt_status = "Worn"
                    self.seatbelt_worn_count += 1
                    self._log_with_timestamp("Simulated: Seatbelt WORN")
                    if self.seatbelt_worn_time is None and self.evaluation_start_time is not None:
                        self.seatbelt_worn_time = time.time()
                        self.calculate_seatbelt_mark()

                if not clean_mode:
                    h, w = frame.shape[:2]
                    x1, y1 = w//4, h//4
                    x2, y2 = 3*w//4, 3*h//4
                    bbox = (x1, y1, x2, y2)
                    color = self.COLOR_GREEN if seatbelt_status == "Worn" else self.COLOR_RED
                    self.draw_bounding_box(frame, x1, y1, x2, y2, color)
                    self.draw_text(frame, x1, y1-10, f"Seatbelt {seatbelt_status} (SIM)", color)
                confidence = 0.8
    
        if seatbelt_status != "Not Detected":
            self.last_seatbelt_status = seatbelt_status

        return frame, seatbelt_status, confidence, bbox  
    
class HeadHandTracker:
    def __init__(self):
        import warnings
        warnings.filterwarnings('ignore')
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        self.seatbelt_detector = SeatbeltDetector()
        
        self.mirror_mark = 0
        self.evaluation_start_time = None
        self.mirror_check_intervals = []  
        self.required_check_interval = 5 
        self.check_window_start = None
        
        self.record_video = False  
        self.video_writer = None
        self.video_filename = None
        
        self.INDEX_FINGER_MCP = 5  
        self.pTime = 0

        self.mirror_left = 0
        self.mirror_right = 0
        self.both_hands_counter = 0
        self.hands_not_detected_counter = 0
        self.straight_wheel_counter = 0
        self.turning_right_counter = 0
        self.turning_left_counter = 0
        
        self.CSV_FILENAME = get_filename_with_id("driver_monitoring", "csv")
        self.csv_file = open(self.CSV_FILENAME, "w", newline="", encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        self.csv_writer.writerow([
            'Timestamp', 'Head_Pose', 'X_Angle', 'Y_Angle', 'Z_Angle', 
            'Hands_Detected', 'Hands_Count', 'Both_Hands_Visible',
            'Hands_Distance_px', 'Hands_Angle_deg', 'Wheel_Position',
            'Face_Detected', 'Frame_Processing_Time_ms', 'Mirror_Check_Recorded'
        ])
        
        self.frame_count = 0
        self.start_time = time.time()
        self.running = False
        self.webcam_cap = None

    def _log_with_timestamp(self, message):
        driver_logger.log_with_timestamp(message)

    def start_evaluation(self):
        self.evaluation_start_time = time.time()
        self.check_window_start = self.evaluation_start_time
        self._log_with_timestamp(f"mirror evaluation started at {datetime.now().strftime('%H:%M:%S')}")
        self._log_with_timestamp(f"expected mirror check interval: every {self.required_check_interval} seconds")

    def calculate_mirror_mark(self):
        actual_checks = self.mirror_left + self.mirror_right
        self._log_with_timestamp(f"DEBUG: mirror counts - L:{self.mirror_left} R:{self.mirror_right} Total:{actual_checks}")
    
        if actual_checks == 0:
            self.mirror_mark = 0
            self._log_with_timestamp("no mirror checks performed --> 0/10")
            return 0

        total_time = time.time() - self.evaluation_start_time
        expected_checks = total_time / self.required_check_interval

        if expected_checks == 0:
            expected_checks = 1

        ratio = (actual_checks / expected_checks) * 100

        self._log_with_timestamp(
            f"mirror check ratio = {ratio:.1f}% "
            f"({actual_checks}/{expected_checks:.1f} expected checks)"
        )

        if ratio <= 25:
            self.mirror_mark = 1
        elif ratio <= 50:
            self.mirror_mark = 3
        elif ratio <= 65:
            self.mirror_mark = 5
        elif ratio <= 79:
            self.mirror_mark = 7
        elif ratio <= 100:
            self.mirror_mark = 10
        elif ratio <= 120:
            self.mirror_mark = 9 
        elif ratio <= 150:
            self.mirror_mark = 7 
        elif ratio <= 200:
            self.mirror_mark = 4  
        else:
            self.mirror_mark = 2 

        if ratio > 100:
            self._log_with_timestamp(f"excessive mirror checking detected: {ratio:.1f}% → {self.mirror_mark}/10")
        else:
            self._log_with_timestamp(f"mirror checking frequency: {ratio:.1f}% → {self.mirror_mark}/10")

        return self.mirror_mark

    def record_mirror_check(self, head_pose):
        current_time = time.time()
        
        if head_pose in ["Looking Left", "Looking Right", "Looking Up"]:
            if (not self.mirror_check_intervals or 
                current_time - self.mirror_check_intervals[-1] >= 2.0):
                self.mirror_check_intervals.append(current_time)
                current_time_str = datetime.now(jordan).strftime("%Y-%m-%d %H:%M:%S")
                self._log_with_timestamp(f"mirror check recorded: {head_pose} at {current_time_str}")
                return True
        return False

    def initialize_zed(self):
        self._log_with_timestamp("Initializing ZED camera")

        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.set_from_serial_number(35674499)
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self._log_with_timestamp(f"ZED open failed: {repr(status)}")
            return False

        self.runtime_params = sl.RuntimeParameters()
        self.image_zed = sl.Mat()

        cam_info = self.zed.get_camera_information()

        try:
            width = cam_info.camera_settings.resolution.width
            height = cam_info.camera_settings.resolution.height
        except AttributeError:
                width, height = 1280, 720

        self._log_with_timestamp(
            f"ZED connected | SN={cam_info.serial_number} | {width}x{height}"
        )

        return True

    def get_zed_frame(self, clean_mode=False):
        if self.zed is None:
            return None

        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None

        self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
        frame = self.image_zed.get_data()[:, :, :3].copy()

        return self.process_frame(frame, clean_mode=clean_mode)
    def get_current_timestamp(self):
        now = datetime.now(jordan)
        return now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def calculate_hands_distance(self, left_hand_landmarks, right_hand_landmarks, img_w, img_h):
        try:
            left_mcp = left_hand_landmarks.landmark[self.INDEX_FINGER_MCP]
            left_x, left_y = left_mcp.x * img_w, left_mcp.y * img_h
            
            right_mcp = right_hand_landmarks.landmark[self.INDEX_FINGER_MCP]
            right_x, right_y = right_mcp.x * img_w, right_mcp.y * img_h
            
            distance = math.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
            return distance
            
        except Exception as e:
            return 0

    def calculate_hands_angle(self, left_hand_landmarks, right_hand_landmarks, img_w, img_h):
        try:
            left_mcp = left_hand_landmarks.landmark[self.INDEX_FINGER_MCP]
            left_x, left_y = left_mcp.x * img_w, left_mcp.y * img_h
            
            right_mcp = right_hand_landmarks.landmark[self.INDEX_FINGER_MCP]
            right_x, right_y = right_mcp.x * img_w, right_mcp.y * img_h
            
            dx = right_x - left_x
            dy = right_y - left_y
            
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            return 0

    def check_both_hands_visible(self, hand_results):
        if not hand_results.multi_hand_landmarks:
            return False, 0
        
        hand_count = len(hand_results.multi_hand_landmarks)
        both_hands_visible = hand_count >= 2
        
        return both_hands_visible, hand_count

    def draw_hands_connection_line(self, image, left_hand_landmarks, right_hand_landmarks, img_w, img_h, clean_mode=False):
        try:
            left_mcp = left_hand_landmarks.landmark[self.INDEX_FINGER_MCP]
            right_mcp = right_hand_landmarks.landmark[self.INDEX_FINGER_MCP]
            
            left_x, left_y = int(left_mcp.x * img_w), int(left_mcp.y * img_h)
            right_x, right_y = int(right_mcp.x * img_w), int(right_mcp.y * img_h)
            
            distance = self.calculate_hands_distance(left_hand_landmarks, right_hand_landmarks, img_w, img_h)
            angle = self.calculate_hands_angle(left_hand_landmarks, right_hand_landmarks, img_w, img_h)
            
            if not clean_mode:
                cv2.line(image, (left_x, left_y), (right_x, right_y), (255, 0, 0), 2)
                cv2.circle(image, (left_x, left_y), 5, (0, 255, 0), -1)
                cv2.circle(image, (right_x, right_y), 5, (0, 0, 255), -1)
                mid_x, mid_y = (left_x + right_x) // 2, (left_y + right_y) // 2
                cv2.putText(image, f"Dist: {distance:.1f} px", (mid_x - 50, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            return distance, angle
            
        except Exception as e:
            return 0, 0
    
    def calculate_steering_mark(self):
        total_frames = self.frame_count
        if total_frames == 0:
            self._log_with_timestamp("no frames processed for steering evaluation --> 0/10")
            return 0

        grip_ratio = self.both_hands_counter / total_frames
        self._log_with_timestamp(f"steering grip ratio = {grip_ratio*100:.1f}% ({self.both_hands_counter}/{total_frames} frames)")

        if grip_ratio >= 0.65:
            self._log_with_timestamp("excellent steering wheel control --> 5/5")
            return 5, grip_ratio
        elif grip_ratio >= 0.35:
            self._log_with_timestamp("moderate steering wheel control --> 2/5")
            return 2, grip_ratio
        else:
            self._log_with_timestamp("poor steering wheel control --> 0/5")
            return 0, grip_ratio

    def log_to_csv(self, data):
        self.csv_writer.writerow(data)
        self.csv_file.flush()  

    def process_frame(self, image, clean_mode=False):
        frame_start_time = time.time()
    
        if not clean_mode:
            image = cv2.flip(image, 1)
        img_h, img_w, img_c = image.shape

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(img_rgb)
        hand_results = self.hands.process(img_rgb)

        image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        timestamp = self.get_current_timestamp()
        head_pose = "Unknown"
        x_angle = 0
        y_angle = 0
        z_angle = 0
        hands_detected = "No"
        hands_count = 0
        both_hands_visible = "No"
        hands_distance = 0
        hands_angle = 0
        wheel_position = "UNKNOWN"
        face_detected = "No"
        mirror_check_recorded = "No"

        seatbelt_image, seatbelt_status, seatbelt_confidence, seatbelt_bbox = \
            self.seatbelt_detector.process_frame(image, clean_mode=clean_mode)
        image = seatbelt_image

        current_time = time.time()
        if (self.seatbelt_detector.evaluation_start_time is not None and 
            current_time - self.seatbelt_detector.evaluation_start_time <= 20 and
            not clean_mode):

            if seatbelt_status == "Worn":
                seatbelt_color = (0, 255, 0)  
            elif seatbelt_status == "Not Worn":
                seatbelt_color = (0, 0, 255)  
            elif seatbelt_status == "Detection Period Ended":
                seatbelt_color = (128, 128, 128) 
            else:
                seatbelt_color = (255, 255, 255) 
        
            if seatbelt_confidence > 0:
                seatbelt_text = f"Seatbelt: {seatbelt_status} ({seatbelt_confidence:.2f})"
            else:
                seatbelt_text = f"Seatbelt: {seatbelt_status}"

            cv2.putText(image, seatbelt_text, (20, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, seatbelt_color, 2)

            elapsed = current_time - self.seatbelt_detector.evaluation_start_time
            time_text = f"Seatbelt Check: {elapsed:.1f}s / 20s"
            cv2.putText(image, time_text, (20, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if face_results.multi_face_landmarks:
            face_detected = "Yes"
            for face_landmarks in face_results.multi_face_landmarks:
                face_2d = []
                face_3d = []

                landmark_indices = [1, 33, 61, 199, 263, 291]

                for idx in landmark_indices:
                    lm = face_landmarks.landmark[idx]
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])       

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                    [0, focal_length, img_h / 2],
                                    [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                if success:
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    x_angle = angles[0] * 360
                    y_angle = angles[1] * 360
                    z_angle = angles[2] * 360

                    DEADZONE_Y = 4
                    DEADZONE_X = 10
                    RIGHT_THRESHOLD_MIN = 1.4
                    RIGHT_THRESHOLD_MAX = 6.5
                    DOWN_THRESHOLD = -8

                    if y_angle < -DEADZONE_Y:
                        head_pose = "Looking Left"
                        self.mirror_left += 1
                    elif y_angle > RIGHT_THRESHOLD_MAX:
                        head_pose = "Looking Backward"
                    elif y_angle > RIGHT_THRESHOLD_MIN:
                        head_pose = "Looking Right"
                        self.mirror_right += 1
                    elif x_angle < DOWN_THRESHOLD:
                        head_pose = "Looking Down"
                    elif x_angle > DEADZONE_X:
                        head_pose = "Looking Up"
                    else:
                        head_pose = "Forward"
                    self.current_head_pose = head_pose

                    if self.evaluation_start_time is not None:
                        if self.record_mirror_check(head_pose):
                            mirror_check_recorded = "Yes"

                    if not clean_mode:
                        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))

                        cv2.putText(image, head_pose, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if hand_results.multi_hand_landmarks:
            hands_detected = "Yes"
            hands_count = len(hand_results.multi_hand_landmarks)
            both_hands_visible_bool, hand_count = self.check_both_hands_visible(hand_results)
            both_hands_visible = "Yes" if both_hands_visible_bool else "No"

            if both_hands_visible_bool:
                self.both_hands_counter += 1
                left_hand = hand_results.multi_hand_landmarks[0]
                right_hand = hand_results.multi_hand_landmarks[1]

                hands_distance, hands_angle = self.draw_hands_connection_line(
                    image, left_hand, right_hand, img_w, img_h, clean_mode=clean_mode
                )

                if abs(hands_angle) < 30:
                    wheel_position = "STRAIGHT"
                    self.straight_wheel_counter += 1
                elif hands_angle > 30:
                    wheel_position = "TURNING RIGHT"
                    self.turning_right_counter += 1
                else:
                    wheel_position = "TURNING LEFT"
                    self.turning_left_counter += 1

                #if not clean_mode:
                    #cv2.putText(image, f"Wheel: {wheel_position}", (20, 150), 
                    #        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    #cv2.putText(image, f"Hands Angle: {hands_angle:.1f} deg", (20, 210), 
                    #       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                #else:
                self.hands_not_detected_counter += 1
        
            if not clean_mode:
                steering_status = "ON STEERING WHEEL" if both_hands_visible_bool else "HANDS NOT DETECTED"
                status_color = (0, 255, 0) if both_hands_visible_bool else (0, 0, 255)

                cv2.putText(image, f"Hands: {hand_count}/2", (20, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        frame_processing_time = (time.time() - frame_start_time) * 1000

        csv_data = [
            timestamp, head_pose, round(x_angle, 2), round(y_angle, 2), round(z_angle, 2),
            hands_detected, hands_count, both_hands_visible,
            round(hands_distance, 2), round(hands_angle, 2), wheel_position,
            face_detected, round(frame_processing_time, 2), mirror_check_recorded
        ]
        self.log_to_csv(csv_data)

        self.frame_count += 1
        return image
    def stop(self):
        self.running = False
        if self.zed is not None:
            self.zed.close()
        time.sleep(2)
        if self.csv_file:
            self.csv_file.close()
            self._log_with_timestamp(f"driver monitoring csv saved: {self.CSV_FILENAME}")

    def generate_driver_mini_report(self):

        mirror_mark = self.calculate_mirror_mark()
        seatbelt_mark = self.seatbelt_detector.calculate_seatbelt_mark()
        steering_mark, grip_ratio = self.calculate_steering_mark()

        total_driver_mark = mirror_mark + seatbelt_mark + steering_mark

        seatbelt_time = "N/A"
        if self.seatbelt_detector.seatbelt_worn_time and self.seatbelt_detector.evaluation_start_time:
            seatbelt_time = f"{self.seatbelt_detector.seatbelt_worn_time - self.seatbelt_detector.evaluation_start_time:.1f} seconds"

        report = f"""MINI-REPORT: DRIVER MONITORING
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
National ID: {NATIONAL_ID}

1. MIRROR CHECKING (Maximum: 10 marks)
   - Left mirror checks: {self.mirror_left}
   - Right mirror checks: {self.mirror_right}
   - Total checks: {self.mirror_left + self.mirror_right}
   - Score: {mirror_mark}/10

2. SEATBELT USAGE (Maximum: 10 marks)
   - Detection window: First 20 seconds
   - Seatbelt worn: {"Yes" if self.seatbelt_detector.seatbelt_worn_time else "No"}
   - Time to wear: {seatbelt_time}
   - Score: {seatbelt_mark}/10

3. STEERING CONTROL (Maximum: 5 marks)
   - Total frames processed: {self.frame_count}
   - Both hands visible frames: {self.both_hands_counter}
   - Steering score: {steering_mark}/5

DRIVER MONITORING TOTAL: {total_driver_mark}/25  

OVERALL ASSESSMENT:
{"PASS" if total_driver_mark >= 15 else "FAIL"}

DRIVER MONITORING TOTAL: {total_driver_mark}/25

"""

        return report


class CrossingDetector:
    def __init__(self):
        self.l1 = 1.76
        self.l2 = 2.76
        self.CHECK_EVERY = 4
        self.YAW_MIN, self.YAW_MAX = 45, 145
        self.PITCH_DEG = 22.0

        self.tz = pytz.timezone("Asia/Amman")

        self._log("Initializing crossing system...")

        self.CSV_FILENAME = get_filename_with_id("crossing_data", "csv")
        self.csv_file = open(self.CSV_FILENAME, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "timestamp_epoch", "timestamp_readable",
            "ty_raw",
            "vel_raw_mps", "vel_raw_kmh",
            "body_id", "pos_x", "pos_y", "pos_z",
            "yaw", "status", "crossing_label"
        ])

        self._log(f"CSV file created: {self.CSV_FILENAME}")

        self.zed = None
        self.prev_pos = {}
        self.prev_ty_raw = None
        self.prev_time = None
        self.frame_count = 0

        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        self.pedestrian_status = {} 
        self.crossing_events = [] 

    def _log(self, message):
        print(f"[CrossingDetector] {message}")

    def _log_crossing_event(self, event_type, pedestrian_id, details=""):
        crossing_logger.log_crossing_event(event_type, pedestrian_id, details)

    def camera_to_global_position(self, px, py, pz):
        angle = math.radians(self.PITCH_DEG)
        Xg = px
        Yg = (py * math.cos(angle)) - (pz * math.sin(angle)) - self.l2
        Zg = (py * math.sin(angle)) + (pz * math.cos(angle)) + self.l1
        return [Xg, Yg, Zg]

    def camera_to_global_quaternion(self, qx, qy, qz, qw):
        angle = math.radians(11)
        xg = (qx * math.cos(angle)) + (qw * math.sin(angle))
        yg = (qy * math.cos(angle)) - (qz * math.sin(angle))
        zg = (qz * math.cos(angle)) + (qy * math.sin(angle))
        wg = (qw * math.cos(angle)) - (qx * math.sin(angle))
        return [xg, yg, zg, wg]

    def draw_bbox(self, frame, obj, status, clean_mode=False):
        if obj.bounding_box_2d is None or clean_mode:
            return frame

        bb = obj.bounding_box_2d
        pts = np.array([(int(p[0]), int(p[1])) for p in bb])
        x1, y1 = np.min(pts, axis=0)
        x2, y2 = np.max(pts, axis=0)

        if status in ["crossing", "about_to_cross"]:
            color = (0, 0, 255)
            label = "CROSSING"
        else:
            color = (0, 255, 0)
            label = "NOT CROSSING"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def initialize_zed_camera(self):
        self._log("Initializing ZED camera...")

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
        init_params.depth_minimum_distance = 2.0
        init_params.depth_maximum_distance = 20.0
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30

        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            self._log("Failed to open ZED camera")
            return False

        track = sl.PositionalTrackingParameters()
        track.set_floor_as_origin = False
        track.enable_imu_fusion = True
        self.zed.enable_positional_tracking(track)

        body_param = sl.BodyTrackingParameters()
        body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
        body_param.enable_tracking = True
        body_param.enable_body_fitting = True
        body_param.body_format = sl.BODY_FORMAT.BODY_34
        self.zed.enable_body_tracking(body_param)

        self._log("ZED camera initialized")
        return True

    def run_combined_mode(self):
        self._log("Starting crossing detection...")

        if not self.initialize_zed_camera():
            return

        runtime = sl.RuntimeParameters()
        runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD


        body_run = sl.BodyTrackingRuntimeParameters()
        body_run.detection_confidence_threshold = 25

        image = sl.Mat()
        bodies = sl.Bodies()
        pose = sl.Pose()

        self.running = True

        while self.running:

            if self.zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            now = datetime.now(self.tz)
            readable = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            epoch = now.timestamp()

            self.zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()[:, :, :3].copy()
            clean_frame = frame.copy()  

            self.zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
            tx_raw, ty_raw, tz_raw = pose.get_translation().get()

            if self.prev_ty_raw is None:
                vel_raw = 0.0
            else:
                dt = epoch - self.prev_time
                vel_raw = (ty_raw - self.prev_ty_raw) / dt if dt > 0 else 0.0

            vel_raw_kmh = vel_raw * 3.6
            self.prev_ty_raw = ty_raw
            self.prev_time = epoch

            self.zed.retrieve_bodies(bodies, body_run)
            wrote = False

            for obj in bodies.body_list:

                if obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK:
                    continue

                px, py, pz = obj.position
                qx, qy, qz, qw = obj.global_root_orientation

                if [qx, qy, qz, qw] == [0, 0, 0, 0]:
                    continue

                Pg = self.camera_to_global_position(px, py, pz)
                qg = self.camera_to_global_quaternion(qx, qy, qz, qw)
                yaw, _, _ = R.from_quat(qg).as_euler("zyx", degrees=True)

                if self.frame_count % self.CHECK_EVERY == 0:
                    old_x = self.prev_pos.get(obj.id, Pg[0])
                    new_x = Pg[0]

                    if self.YAW_MIN <= abs(yaw) <= self.YAW_MAX:
                        if (old_x < 0 < new_x) or (old_x > 0 > new_x):
                            status = "crossing"
                            if obj.id not in self.pedestrian_status or self.pedestrian_status[obj.id] != "crossing":
                                self._log_crossing_event("PEDESTRIAN STARTED CROSSING", obj.id, 
                                                       f"Position: x={Pg[0]:.2f}, y={Pg[1]:.2f}, z={Pg[2]:.2f}, Yaw={yaw:.1f}°")
                                self.pedestrian_status[obj.id] = "crossing"
                        else:
                            status = "about_to_cross"
                            if obj.id not in self.pedestrian_status or self.pedestrian_status[obj.id] != "about_to_cross":
                                self._log_crossing_event("PEDESTRIAN ABOUT TO CROSS", obj.id,
                                                       f"Position: x={Pg[0]:.2f}, yaw={yaw:.1f}°, Velocity={vel_raw_kmh:.1f} km/h")
                                self.pedestrian_status[obj.id] = "about_to_cross"
                    else:
                        status = "not_crossing"
                        if obj.id in self.pedestrian_status and self.pedestrian_status[obj.id] != "not_crossing":
                            self._log_crossing_event("PEDESTRIAN NOT CROSSING", obj.id,
                                                   f"Yaw={yaw:.1f}° (outside crossing range)")
                            self.pedestrian_status[obj.id] = "not_crossing"

                    self.prev_pos[obj.id] = new_x
                else:
                    status = "not_crossing"

                crossing_label = status

                frame = self.draw_bbox(frame, obj, status, clean_mode=False)

                self.csv_writer.writerow([
                    epoch, readable,
                    ty_raw,
                    vel_raw, vel_raw_kmh,
                    obj.id,
                    Pg[0], Pg[1], Pg[2],
                    yaw,
                    status,
                    crossing_label
                ])
                self.csv_file.flush()
                wrote = True

            if not wrote:
                self.csv_writer.writerow([
                    epoch, readable,
                    ty_raw,
                    vel_raw, vel_raw_kmh,
                    "none",
                    "none", "none", "none",
                    "none",
                    "no_person",
                    "none"
                ])
                self.csv_file.flush()
                
                if self.pedestrian_status:
                    self._log_crossing_event("NO PEDESTRIANS DETECTED", "N/A", 
                                           f"Previously tracking {len(self.pedestrian_status)} pedestrian(s)")
                    self.pedestrian_status.clear()

            with self.frame_lock:
                self.current_frame = frame.copy()
                self.current_clean_frame = clean_frame.copy()

            self.frame_count += 1

        self.zed.close()
        self.csv_file.close()
        self._log(f"Crossing detection stopped. CSV saved: {self.CSV_FILENAME}")
        self._log_crossing_event("SYSTEM STOPPED", "N/A", f"Total frames processed: {self.frame_count}")

    def get_current_frame(self, clean_mode=False):
        with self.frame_lock:
            if clean_mode:
                return self.current_clean_frame.copy() if hasattr(self, 'current_clean_frame') and self.current_clean_frame is not None else None
            else:
                return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        self._log("Stopping crossing detection...")
        self.running = False
        time.sleep(1)
        self._log("Stopped.")

    def generate_crossing_mini_report(self, crossing_mark, crossing_decision, crossing_details):
        report = f"""MINI-REPORT: PEDESTRIAN CROSSING BEHAVIOR
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
National ID: {NATIONAL_ID}

OBJECTIVE: Assess driver's response to pedestrian crossings

REQUIREMENT:
- Slow down or stop appropriately before pedestrian crossings
- Maintain safe speed when pedestrians are crossing or about to cross

YOUR PERFORMANCE:
- Your Score: {crossing_mark}/25
- Decision: {crossing_decision}
- Total crossing events analyzed: {len(crossing_details)}

CROSSING EVENT ANALYSIS:
"""
        
        if crossing_details:
            for i, c in enumerate(crossing_details, 1):
                report += f"""
{i}. Event #{c['id']} at {c['time']}
   - Score: {c['score']}/25
   - Decision: {c['decision']}
   - Justification: {c['justification']}
"""
        else:
            report += "   No crossing events detected during evaluation period.\n"
        
        report += f"""
DETAILED ASSESSMENT:
- Expected behavior: Decelerate when pedestrians are about to cross
- Your response: {'Appropriate deceleration observed' if crossing_mark >= 20 else 'Inconsistent deceleration' if crossing_mark >= 12 else 'Insufficient deceleration'}
- Safety level: {'Safe' if crossing_mark >= 20 else 'Moderate' if crossing_mark >= 12 else 'Unsafe'}

RECOMMENDATIONS:
{'- Maintain good crossing awareness and braking response' if crossing_mark >= 20 else 
 '- Improve reaction time and deceleration before pedestrian crossings' if crossing_mark >= 12 else 
 '- Urgent improvement needed: Practice defensive driving near pedestrian crossings'}

CROSSING BEHAVIOR ASSESSMENT: {'PASS' if crossing_mark >= 12 else 'FAIL'}
"""
        return report

class IntegratedMonitoringSystem:
    def __init__(self):
        self.crossing_detector = CrossingDetector()
        self.head_hand_tracker = HeadHandTracker()
        self.running = False
        self.annotated_writer = None
        self.clean_writer = None
        #self.annotated_window_name = "Driver & Crossing Monitoring"
        self.evaluation_start_time = None

    def setup_annotated_recording(self, width, height, fps=30):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.annotated_filename = get_filename_with_id("annotated_monitoring", "avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.annotated_writer = cv2.VideoWriter(self.annotated_filename, fourcc, fps, (width, height))
        print(f"Annotated video recording: {os.path.basename(self.annotated_filename)}")
        
        self.clean_filename = get_filename_with_id("clean_monitoring", "avi")
        self.clean_writer = cv2.VideoWriter(self.clean_filename, fourcc, fps, (width, height))
        print(f"Clean video recording: {os.path.basename(self.clean_filename)}")

    def combine_feeds(self, driver_frame, crossing_frame, clean_mode=False):
        target_height = 360
        
        if clean_mode:
            driver_clean = self.head_hand_tracker.get_zed_frame(clean_mode=True) if hasattr(self.head_hand_tracker, 'get_zed_frame') else driver_frame
            crossing_clean = self.crossing_detector.get_current_frame(clean_mode=True)
            
            if driver_clean is None:
                driver_clean = driver_frame
            if crossing_clean is None:
                crossing_clean = crossing_frame
            
            driver_resized = cv2.resize(driver_clean, (640, target_height))
            crossing_resized = cv2.resize(crossing_clean, (640, target_height))
        else:
            driver_resized = cv2.resize(driver_frame, (640, target_height))
            crossing_resized = cv2.resize(crossing_frame, (640, target_height))
        
        annotated_frame = np.hstack((driver_resized, crossing_resized))
        
        if not clean_mode:
            cv2.line(annotated_frame, (640, 0), (640, target_height), (255, 255, 255), 2)
        
        if self.evaluation_start_time is not None and not clean_mode:
            elapsed_time = time.time() - self.evaluation_start_time
            
        
            
            # Show frame counter
            #if hasattr(self.head_hand_tracker, 'frame_count'):
            #    cv2.putText(annotated_frame, f"Frame: {self.head_hand_tracker.frame_count}", (10, 100), 
            #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame

    def run_combined_system(self):
        print("Starting Combined Monitoring System")
        print(f"National ID: {NATIONAL_ID}")
        print(f"Output Directory: {BASE_DIR}")
        print("Camera Configuration:")
        print("/tLeft: Driver Monitoring (ZED)")
        print("/tRight: Crossing Detection (ZED)")
        print("/tCombined display + recording")
        print("/tSaving both annotated and clean video feeds")
        
        print("\nCONTROLS:")
        print("  Press 'ESC' to end the entire evaluation")
        
        print("\ninit cameras")
        
        crossing_thread = threading.Thread(target=self.crossing_detector.run_combined_mode)
        crossing_thread.daemon = True
        crossing_thread.start()
        
        time.sleep(5) 
      
        
        if not self.head_hand_tracker.initialize_zed():
            print("Driver ZED failed")
            return
        
        print("all cams initialized")
        
        self.setup_annotated_recording(1280, 360)  
        
        self.evaluation_start_time = time.time()
        self.head_hand_tracker.start_evaluation()
        self.head_hand_tracker.seatbelt_detector.start_evaluation()
        
        print("evaluation started")
        print("evaluation Criteria:")
        print("/tmirror checks (Left/Right/Up): Every 4-6 seconds")
        print("/tseatbelt: Must be worn within 20 seconds (detection stops after 20s)")
        
        self.running = True
        print("\nstarting combined monitoring...")
        
        try:
            frame_count = 0
            while self.running:
                driver_frame = self.head_hand_tracker.get_zed_frame(clean_mode=False)
                if driver_frame is None:
                    print("no driver frame")
                    break
                
                crossing_frame = self.crossing_detector.get_current_frame(clean_mode=False)
                if crossing_frame is None:
                    crossing_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(crossing_frame, "NO ZED FRAME", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                combined_annotated = self.combine_feeds(driver_frame, crossing_frame, clean_mode=False)
                
                combined_clean = self.combine_feeds(driver_frame, crossing_frame, clean_mode=True)
                
                # Display annotated version
                #cv2.imshow(self.annotated_window_name, combined_annotated)
                
                if self.annotated_writer is not None:
                    self.annotated_writer.write(combined_annotated)
                if self.clean_writer is not None:
                    self.clean_writer.write(combined_clean)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed_time = time.time() - self.evaluation_start_time
                    print(f"Frames processed: {frame_count}, Running time: {elapsed_time:.1f}s")
                    
        except Exception as e:
            print(f"error in combined system: {e}")
        finally:
            self.stop_combined_system()
            print("combined monitoring system stopped")

    def evaluate_crossing_csv(self):
        df = pd.read_csv(self.crossing_detector.CSV_FILENAME)
        df["time_dt"] = pd.to_datetime(df["timestamp_readable"])

        df["vel_raw_kmh_smooth"] = df["vel_raw_kmh"].rolling(window=20, center=True).mean()

        def speed_category(v):
            avg = v.mean()
            if avg < 2: return "Stopped"
            elif avg < 10: return "Very Slow"
            else: return "Normal"

        def evaluate(slope_label, speed_label):
            # increasing
            if slope_label == "Increasing" and speed_label in ["Normal", "Very Slow"]:
                return 0, "FAIL"
            if slope_label == "Increasing" and speed_label == "Stopped":
                return 25, "PASS"

            # decreasing
            if slope_label == "Decreasing" and speed_label in ["Very Slow", "Stopped"]:
                return 25, "PASS"
            if slope_label == "Decreasing" and speed_label == "Normal":
                return 24, "PASS"

            # flat
            if slope_label == "Flat" and speed_label == "Stopped":
                return 25, "PASS"
            if slope_label == "Flat" and speed_label == "Very Slow":
                return 22.5, "PASS"
            if slope_label == "Flat" and speed_label == "Normal":
                return 12.5, "FAIL"

            return 0, "FAIL"

        crossing_details = []
        cross_times = df.loc[df["crossing_label"] == "crossing", "time_dt"]

        for t in cross_times:
            seg = df[(df["time_dt"] >= t - pd.Timedelta(seconds=3)) &
                     (df["time_dt"] <= t)]

            if len(seg) < 3:
                continue

            tB = (seg["time_dt"] - seg["time_dt"].iloc[0]).dt.total_seconds()
            sB = seg["vel_raw_kmh_smooth"]

            slope, intercept = np.polyfit(tB, sB, 1)

            if slope > 0.05:
                slope_label = "Increasing"
            elif slope < -0.05:
                slope_label = "Decreasing"
            else:
                slope_label = "Flat"

            speed_label = speed_category(sB)
            speed_avg = sB.mean()

            score, decision = evaluate(slope_label, speed_label)

            crossing_details.append({
                "id": df.loc[df["time_dt"] == t, "body_id"].values[0],
                "time": t.strftime("%H:%M:%S.%f")[:-3],
                "score": score,
                "decision": decision,
                "justification": f"Speed={speed_avg:.2f} km/h ({speed_label}) and {slope_label}"
            })

        failing = [c for c in crossing_details if c["decision"] == "FAIL"]

        final_score = 0 if len(failing) > 0 else 25
        final_decision = "FAIL" if len(failing) > 0 else "PASS"

        return crossing_details, final_score, final_decision
    
    def generate_mini_reports(self):
        print("\n" + "="*60)
        print("GENERATING MINI-REPORTS")
        print("="*60)
        
        driver_report = self.head_hand_tracker.generate_driver_mini_report()
        driver_filename = get_filename_with_id("driver_mini_report", "txt")
        with open(driver_filename, 'w') as f:
            f.write(driver_report)
        print(f"Driver monitoring mini-report saved: {os.path.basename(driver_filename)}")
        print(driver_report)
        
        cross_details, crossing_mark, crossing_decision = self.evaluate_crossing_csv()
        crossing_report = self.crossing_detector.generate_crossing_mini_report(
            crossing_mark, crossing_decision, cross_details
        )
        crossing_filename = get_filename_with_id("crossing_mini_report", "txt")
        with open(crossing_filename, 'w') as f:
            f.write(crossing_report)
        print(f"Crossing mini-report saved: {os.path.basename(crossing_filename)}")
        print(crossing_report)
        
       
           
        return driver_report, crossing_report

    def generate_final_report(self):
        print("\n" + "="*60)
        print("GENERATING FINAL EVALUATION REPORT")
        print("="*60)
        
        driver_report, crossing_report = self.generate_mini_reports()
        
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE FINAL REPORT")
        print("="*60)

        cross_details, crossing_mark, crossing_decision = self.evaluate_crossing_csv()

        mirror_mark = self.head_hand_tracker.calculate_mirror_mark()
        seatbelt_mark = self.head_hand_tracker.seatbelt_detector.calculate_seatbelt_mark()

        steering_mark, grip_ratio = self.head_hand_tracker.calculate_steering_mark()
        total_mark = mirror_mark + seatbelt_mark + steering_mark + crossing_mark
        total_evaluation_time = time.time() - self.evaluation_start_time

        report_content = f"""COMPREHENSIVE DRIVER EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
National ID: {NATIONAL_ID}
Evaluation Duration: {total_evaluation_time:.1f} seconds

SYSTEM OVERVIEW:
================
This comprehensive evaluation assesses three critical driving competencies:
1. Driver Monitoring (Mirror checks, seatbelt usage, steering control)
2. Pedestrian Crossing Behavior (Response to crossing pedestrians)

Note: Detailed mini-reports for each segment have been generated separately.

EVALUATION RESULTS:
==================

1. DRIVER MONITORING (Maximum: 20 marks)
   A. Mirror Checking (10 marks)
      - Mirror checks recorded: {len(self.head_hand_tracker.mirror_check_intervals)}
      - Your Score: {mirror_mark}/10
      - Assessment: {'Excellent (≥80%)' if mirror_mark >= 8 else 'Good (60-79%)' if mirror_mark >= 5 else 'Insufficient (<60%)'}

   B. Seatbelt Usage (10 marks)
      - Detection period: First 20 seconds only
      - Seatbelt worn: {'Yes' if self.head_hand_tracker.seatbelt_detector.seatbelt_worn_time else 'No'}
      - Your Score: {seatbelt_mark}/10

    C. Steering Control (5 marks)
      - Grip ratio: {grip_ratio:.1%}
      - Your Score: {steering_mark}/5

   D. Driver Monitoring Total: {mirror_mark + seatbelt_mark + steering_mark}/25

2. PEDESTRIAN CROSSING BEHAVIOR (Maximum: 25 marks)
   - Your Score: {crossing_mark}/25
   - Decision: {crossing_decision}
   - Crossing events analyzed: {len(cross_details)}

   Crossing Event Breakdown:
{''.join([f"     • ID {c['id']} at {c['time']} → {c['score']}/25 | {c['justification']}" for c in cross_details])}

FINAL SCORE SUMMARY:
====================
- Driver Monitoring: {mirror_mark + seatbelt_mark + steering_mark}/25
- Pedestrian Crossing: {crossing_mark}/25
- TOTAL SCORE: {total_mark}/50

OVERALL ASSESSMENT: {'PASS' if total_mark >= 30 else 'FAIL'}

KEY RECOMMENDATIONS:
====================
{'- Continue good mirror checking habits' if mirror_mark >= 8 else '- Increase frequency of mirror checks' if mirror_mark >= 5 else '- Significantly improve mirror checking frequency'}
{'- Good seatbelt usage timing' if seatbelt_mark == 10 else '- Wear seatbelt earlier (within first 20 seconds)'}
{'- Excellent braking response at pedestrian crossings' if crossing_mark >= 20 else
 '- Improve deceleration and reaction time before crossing pedestrians' if crossing_mark >= 12 else
 '- Unsafe behavior at pedestrian crossings — immediate improvement required'}

DATA FILES GENERATED:
=====================
- National ID Directory: {BASE_DIR}
- Annotated video: {os.path.basename(getattr(self, 'annotated_filename', 'Not saved'))}
- Clean video (no annotations): {os.path.basename(getattr(self, 'clean_filename', 'Not saved'))}
- Driver monitoring CSV: {os.path.basename(self.head_hand_tracker.CSV_FILENAME)}
- Crossing detection CSV: {os.path.basename(self.crossing_detector.CSV_FILENAME)}
- Mini-reports for each evaluation segment (separate files)

FINAL VERDICT:
==============
Based on the comprehensive evaluation, you have demonstrated {'satisfactory' if total_mark >= 25 else 'developing' if total_mark >= 15 else 'insufficient'} 
driving competency across all assessed areas.

Generated by Integrated Driver Monitoring System
"""
        
        report_filename = get_filename_with_id("comprehensive_final_report", "txt")
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("\n" + "="*50)
        print("FINAL REPORT SUMMARY")
        print("="*50)
        print(f"National ID: {NATIONAL_ID}")
        print(f"Total Score: {total_mark}/50")
        print(f"Overall Assessment: {'PASS' if total_mark >= 25 else 'NEEDS IMPROVEMENT' if total_mark >= 15 else 'FAIL'}")
        print(f"Comprehensive report saved to: {os.path.basename(report_filename)}")
        print(f"All files saved in directory: {BASE_DIR}")
        
        print("\nQUICK SCORE SUMMARY:")
        print(f"  Driver Monitoring: {mirror_mark + seatbelt_mark}/20")
        print(f"  Pedestrian Crossing: {crossing_mark}/25")
        print(f"  TOTAL: {total_mark}/50")
        return report_content, total_mark

    def stop_combined_system(self):
        print("\nStopping combined monitoring system...")
        if hasattr(self, '_stop_called') and self._stop_called:
            print("Stop already called, skipping duplicate call")
            return
        self._stop_called = True  
    
        print("\nStopping combined monitoring system...")
        self.running = False

        if self.annotated_writer is not None:
            self.annotated_writer.release()
            print(f"Annotated AVI saved: {os.path.basename(self.annotated_filename)}")

        if self.clean_writer is not None:
            self.clean_writer.release()
            print(f"Clean AVI saved: {os.path.basename(self.clean_filename)}")
        time.sleep(5)
        self.annotated_mp4 = None
        self.clean_mp4 = None

        try:
            if hasattr(self, "annotated_filename") and os.path.exists(self.annotated_filename):
                self.annotated_mp4 = convert_avi_to_mp4_slow(self.annotated_filename)
                print(f"Annotated MP4 ready: {os.path.basename(self.annotated_mp4)}")

            if hasattr(self, "clean_filename") and os.path.exists(self.clean_filename):
                self.clean_mp4 = convert_avi_to_mp4_slow(self.clean_filename)
                print(f"Clean MP4 ready: {os.path.basename(self.clean_mp4)}")

        except Exception as e:
            print(f"MP4 conversion failed: {e}")

        try:
            self.crossing_detector.stop()
        except Exception as e:
            print(f"Crossing detector stop error: {e}")

        try:
            self.head_hand_tracker.stop()
        except Exception as e:
            print(f"Driver tracker stop error: {e}")

        cv2.destroyAllWindows()

        print("System stopped successfully")
        print(f"All outputs saved in: {BASE_DIR}")



def main():
    print("COMPLETE DRIVER MONITORING & EVALUATION SYSTEM")
    print("=" * 50)
    
    national_id = get_national_id()
    
    init_loggers()
    
    print(f"\nStarting evaluation for National ID: {national_id}")
    print(f"All files will be saved in: {BASE_DIR}")
    print("\nFeatures:")
    print("- Driver monitoring with mirror and seatbelt detection")
    print("- Pedestrian crossing detection with event logging")
    print("- Combined display with real-time annotations")
    print("- Saves both annotated and clean video feeds")
    print("- Generates mini-reports for each segment")
    print("- Generates comprehensive final evaluation report")
    print("\nNote: Seatbelt detection is active only for the first 20 seconds of the test.")
    
    integrated_system = IntegratedMonitoringSystem()
    
    try:
        integrated_system.run_combined_system()
    except KeyboardInterrupt:
        print("\nsystem stopped by keyboard interrupt")
        integrated_system.generate_final_report()
        integrated_system.stop_combined_system()


if __name__ == "__main__":
    main()
