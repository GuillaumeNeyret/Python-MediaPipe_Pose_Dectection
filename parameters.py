"All parameters all here "
import cv2
from enum import Enum
# import pyautogui

# Camera resolution
res_cam_height = 1080   # 2160 for 4K Cam
res_cam_width = 1920    # 3840 for 4K Cam

# crop_height = min(window_height//2,res_cam_height)  # //2 Because we want to display 2 img on the same screen
# crop_width = min(window_width,res_cam_width)        # Minimum to adjust if cam res < screen res

crop_height = res_cam_height  # //2 Because we want to display 2 img on the same screen
crop_width = int(res_cam_height*9/16)        # Minimum to adjust if cam res < screen res


crop_dim = (crop_height, crop_width)
center = (int(res_cam_height*0.5), res_cam_width//2) 

# MediaPipe Holistic Settings
settings = {
    # 'static_image_mode': False,
    'model_complexity': 0,
    # 'smooth_landmarks': True,
    # 'enable_segmentation': False,
    # 'smooth_segmentation': True,
    # 'refine_face_landmarks': False,
    'min_detection_confidence': 0.6,
    'min_tracking_confidence': 0.3
}

# MediaPipe Holistic Settings
settings_face_mesh = {
    'static_image_mode' : False,
    'max_num_faces' : 2,
    'refine_landmarks' : False,
    'min_detection_confidence' : 0.5,
    'min_tracking_confidence' : 0.5
}


# MP Landmarks Drawing settings
draw_face_landmark = {
    'color' : (0, 128, 200),  #BGR color
    'thickness' : 0,
    'circle_radius' : 1
}
draw_face_connection = {
    'color' : (0, 0, 255),
    'thickness' : 2,
    'circle_radius' : 1
}

draw_hand_landmark = {
    'color' : (255, 0, 0),
    'thickness' : 1,
    'circle_radius' : 2
}
draw_hand_connection = {
    'color' : (255, 0, 0),
    'thickness' : 1,
    'circle_radius' : 0
}

draw_body_landmark = {
    'color' : (0, 0, 255),
    'thickness' : 0,
    'circle_radius' : 4
}
draw_body_connection = {
    'color' : (0, 255, 0),
    'thickness' : 2,
    'circle_radius' : 0
}


font = cv2.FONT_HERSHEY_SIMPLEX

# List of values needed from Blendshape
blend_list = [
    "JAW_OPEN",
    "BROW_DOWN_LEFT",
    "BROW_DOWN_RIGHT",
    "BROW_OUTER_UP_LEFT",
    "BROW_OUTER_UP_RIGHT",
    "MOUTH_CLOSE",
    "MOUTH_PUCKER",
    "EYE_BLINK_LEFT",
    "EYE_BLINK_RIGHT",
    "MOUTH_SMILE_LEFT",
    "MOUTH_SMILE_RIGHT"
    ]

class event_triggers(float, Enum):
    NO_BLINK = 0.2
    BLINK = 0.3
    KISS = 0.4
    BROW_DOWN = 0.15
    OPEN_MOUTH = 0.5
    NO_OPEN_MOUTH = 0.1
    SMILE = 0.5
    BROW_UP = 0.3
    BLINK_DIFF = 0.1