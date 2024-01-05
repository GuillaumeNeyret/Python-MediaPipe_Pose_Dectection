"All parameters all here "
import cv2
# import pyautogui

# Camera resolution
res_cam_height = 1080   # 2160 for 4K Cam
res_cam_width = 1920    # 3840 for 4K Cam

# MediaPipe Holistic Settings
settings = {
    # 'static_image_mode': False,
    'model_complexity': 1,
    # 'smooth_landmarks': True,
    # 'enable_segmentation': False,
    # 'smooth_segmentation': True,
    # 'refine_face_landmarks': False,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# MP Landmarks Drawing settings
draw_face_landmark = {
    'color' : (0, 255, 0),
    'thickness' : 0,
    'circle_radius' : 0
}
draw_face_connection = {
    'color' : (0, 0, 255),
    'thickness' : 1,
    'circle_radius' : 0
}

draw_hand_landmark = {
    'color' : (255, 0, 0),
    'thickness' : 0,
    'circle_radius' : 0
}
draw_hand_connection = {
    'color' : (255, 0, 0),
    'thickness' : 1,
    'circle_radius' : 0
}

draw_body_landmark = {
    'color' : (0, 0, 255),
    'thickness' : 0,
    'circle_radius' : 2
}
draw_body_connection = {
    'color' : (0, 255, 0),
    'thickness' : 2,
    'circle_radius' : 0
}

# Test mode
mode_img = 'img'
mode_cam = 'cam'
mode = mode_cam

# window_width, window_height = pyautogui.size()

# crop_height = min(window_height//2,res_cam_height)  # //2 Because we want to display 2 img on the same screen
# crop_width = min(window_width,res_cam_width)        # Minimum to adjust if cam res < screen res
# # crop_width = window_width
# # if crop_height>res_cam_height:
# #     crop_height=res_cam_height
# #
# # if crop_width>res_cam_width:
# #     crop_width=res_cam_width

# crop_dim = (crop_height, crop_width)
# center = (int(res_cam_height*0.5), res_cam_width//2)       # Crop Center // Adjust height

# Font Display Settings
font = cv2.FONT_HERSHEY_SIMPLEX