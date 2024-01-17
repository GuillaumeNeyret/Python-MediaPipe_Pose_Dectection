import mediapipe as mp
from parameters import *
import time, cv2
from mediapipe.tasks.python.vision.face_landmarker import Blendshapes
from typing import List, Dict

"""
==========================================================================================================
FUNCTIONS DEFINITION
==========================================================================================================
"""




"""
==========================================================================================================
VARIABLES
==========================================================================================================
"""

# CAMERA
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set Camera Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_cam_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_cam_height)

prev_frame_time = 0
err = 0
timestamp = 0
hands_gesture = {
    "LEFT":None,
    "RIGHT": None
}

"""
==========================================================================================================
MODELS INITIALIZATION (MP TASK)
==========================================================================================================
"""

# HOLISTIC & DRAWING
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic                 # Load Holistic module

# SHARED VAR FOR MODELS 
BaseOptions = mp.tasks.BaseOptions                  
VisionRunningMode = mp.tasks.vision.RunningMode

# HAND MODEL
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    category_name = result.gestures[0][0].category_name
    # print(result.gestures)
    # print(category_name)
    # global gesture= result.gestures[0][0].category_name
    # return

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='assets/models/custom_model.task'),
    running_mode=VisionRunningMode.IMAGE
    )

recognizer = GestureRecognizer.create_from_options(options)



# Initializes holistic model
with mp_holistic.Holistic(**settings) as holistic :      # Create holistic object
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame ...")
            err += 1
            if err == 3:                                 # 3 Consecutive unreadable frame stop the process
                break
            continue

        err = 0
        timestamp += 1

        image = frame

        """
        ==========================================================================================================
        HOLISTIC PART
        ==========================================================================================================
        """

        # Recolor Feed into RGB for MP
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)


        # Draw landmarks
        # Hands
        """ ===========================
                    RIGHT HAND
        =============================== """ 
        if results.right_hand_landmarks:
            # Gets Hand coords
            right_hand_landmarks = results.right_hand_landmarks.landmark
            x_min = int(min(right_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*0.90)
            x_max = int(max(right_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*1.1)
            y_min = int(min(right_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*0.90)
            y_max = int(max(right_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*1.1)
            # Extract ROI
            ROI = image[y_min:y_max, x_min:x_max]
            right_hand_frame = ROI.copy()
            

            # Draw landmarks
            mp_drawing.draw_landmarks(image=image,
                                    landmark_list=results.right_hand_landmarks,
                                    connections=mp_holistic.HAND_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_landmark),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_connection)
                                    )
            
            # GESTURE RECOGNITION
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=right_hand_frame)
            Hand_results = recognizer.recognize(mp_image)
            if Hand_results.gestures :
                hands_gesture['RIGHT'] = Hand_results.gestures[0][0].category_name
            else:
                hands_gesture['RIGHT'] = None
        else :
            hands_gesture['RIGHT'] = None
            
        """ ===========================
                    LEFT HAND
        =============================== """ 
        if results.left_hand_landmarks:
            # Gets Hand coords
            left_hand_landmarks = results.left_hand_landmarks.landmark
            x_min = int(min(left_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*0.90)
            x_max = int(max(left_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*1.1)
            y_min = int(min(left_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*0.90)
            y_max = int(max(left_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*1.1)
            # Extract ROI
            ROI = image[y_min:y_max, x_min:x_max]
            left_hand_frame = ROI.copy()
            # Draw landmarks
            mp_drawing.draw_landmarks(image=image,
                                    landmark_list=results.left_hand_landmarks,
                                    connections=mp_holistic.HAND_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_landmark),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_connection)
                                    )
            
            # GESTURE RECOGNITION
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=left_hand_frame)
            Hand_results = recognizer.recognize(mp_image)
            if Hand_results.gestures :
                hands_gesture['LEFT'] = Hand_results.gestures[0][0].category_name if (Hand_results.gestures[0][0].category_name != '' and Hand_results.gestures[0][0].category_name != 'none')  else None
            else:
                hands_gesture['LEFT'] = None
        else :
            hands_gesture['LEFT'] = None
        
        """
        ==========================================================================================================
        DISPLAY
        ==========================================================================================================
        """  
        # Recolor Feed into BGR for Display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image,1)

        # Display hand frame
        # if results.right_hand_landmarks:
        #     right_hand_frame = cv2.cvtColor(right_hand_frame, cv2.COLOR_RGB2BGR)
        #     right_hand_frame = cv2.flip(right_hand_frame,1)
        #     cv2.imshow('RIGHT HAND', right_hand_frame)
        
        # if results.left_hand_landmarks:
        #     left_hand_frame = cv2.cvtColor(left_hand_frame, cv2.COLOR_RGB2BGR)
        #     left_hand_frame = cv2.flip(left_hand_frame,1)
        #     cv2.imshow('LEFT HAND', left_hand_frame)

        for event_name, event_state in hands_gesture.items():
            text = f'{event_name}: {event_state}'
            color = (0, 255, 0)
            cv2.putText(image, text, (10, 30 * (1 + list(hands_gesture.keys()).index(event_name))),font, 1, color, 2, cv2.LINE_AA)


        # FPS Display
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        cv2.putText(image, str(fps), (500, 500), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


        # Display the resulting frame
        cv2.imshow('MP TEST', image)

        # Press 'q' or 'Esc" to exit
        if cv2.waitKey(5) & 0xFF in [ord('q'), 27]:
            break

cam.release()
cv2.destroyAllWindows()
