import mediapipe as mp
from parameters import *
import time, cv2
from mediapipe.tasks.python.vision.face_landmarker import Blendshapes
from typing import List, Dict
from margins import Margins

"""
==========================================================================================================
FUNCTIONS DEFINITION
==========================================================================================================
"""

# Transform a blendshape to a dict with only values from the need_value list
def blendshapes_to_dict(face_blendshape):

    return {i : face_blendshape[getattr(Blendshapes, i).value].score for i in blend_list}

# Returns True if Kiss 
def kiss(blend_values:Dict[str,float]) -> bool:
    if blend_values['MOUTH_PUCKER']> event_triggers.KISS and blend_values["JAW_OPEN"]<event_triggers.NO_OPEN_MOUTH:
        return True
    return False

# Returns True if Left Blink only
def left_blink(blend_values:Dict[str,float])-> bool:
    if blend_values['EYE_BLINK_LEFT']>event_triggers.BLINK and blend_values['EYE_BLINK_RIGHT']<event_triggers.NO_BLINK:
        return True
    return False

# Returns True if Smile
def smile(blend_values:Dict[str,float])-> bool:
    if blend_values['MOUTH_SMILE_LEFT']>event_triggers.SMILE and blend_values['MOUTH_SMILE_LEFT']>event_triggers.SMILE:
        return True
    return False

# Returns True if surprise
def surprise(blend_values:Dict[str,float])-> bool:
    if (blend_values["BROW_OUTER_UP_LEFT"]>event_triggers.BROW_UP and blend_values["BROW_OUTER_UP_RIGHT"]>event_triggers.BROW_UP and 
        blend_values['EYE_BLINK_LEFT']<event_triggers.NO_BLINK and blend_values['EYE_BLINK_RIGHT']<event_triggers.NO_BLINK):
        return True
    return False


def angry(blend_values:Dict[str,float])-> bool:
    if blend_values['BROW_DOWN_LEFT']>event_triggers.BROW_DOWN and blend_values['BROW_DOWN_RIGHT']>event_triggers.BROW_DOWN:
        return True
    return False

# Return face expression as a dict
def event_faces(blend_values: Dict[str, float]) -> Dict[str, bool]:
    return {
        'smile': smile(blend_values),
        'kiss': kiss(blend_values),
        'left_blink': left_blink(blend_values),
        'surprise': surprise(blend_values),
        'angry': angry(blend_values)
    }



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

margin_hand = Margins(top=0.1, right=0.1, bottom=0.1, left=0.1)


face_status={
    'smile': None,
    'kiss': None,
    'left_blink': None,
    'surprise': None,
    'angry': None
    }

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

# FACIAL MODEL
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='assets/models/face_landmarker_v2_with_blendshapes.task'),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
    )

FaceRecognizer = FaceLandmarker.create_from_options(options)

# HAND MODEL
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    category_name = result.gestures[0][0].category_name

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='assets/models/custom_model.task'),
    running_mode=VisionRunningMode.IMAGE
    )

HandRecognizer = GestureRecognizer.create_from_options(options)

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

        """ ===========================
                    FACE
        =============================== """ 
        if results.face_landmarks :

            # Gets Face coords
            head_landmarks = results.face_landmarks.landmark
            x_min = int(min(head_landmarks, key=lambda x: x.x).x * image.shape[1])
            x_max = int(max(head_landmarks, key=lambda x: x.x).x * image.shape[1])
            y_min = int(min(head_landmarks, key=lambda y: y.y).y * image.shape[0])
            y_max = int(max(head_landmarks, key=lambda y: y.y).y * image.shape[0])
            # Extract ROI
            ROI = image[y_min:y_max, x_min:x_max]
            face_frame = ROI.copy()

            # FACE BELNDSHAPES RECOGNIZER
            mp_Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame)
            face_result = FaceRecognizer.detect_for_video(mp_Image,timestamp)

            if face_result != None:
                face_values = blendshapes_to_dict(face_blendshape=face_result.face_blendshapes[0]) # get all needed values from blend list
                # print (face_values)
                face_status = event_faces(face_values)
            else:
                face_status = {key: None for key in face_status}   
                

            # Draw landmarks
            mp_drawing.draw_landmarks(image=image,
                                        landmark_list=results.face_landmarks,
                                        connections= mp_holistic.FACEMESH_CONTOURS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_face_landmark),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(**draw_face_connection)
                                        )
            
            # Display face results  
            for event_name, event_state in face_status.items():
                text = f'{event_name}: {event_state}'
                color = (0, 255, 0) if event_state else (0, 0, 255)
                cv2.putText(image, text, (10, 30 * (1 + list(face_status.keys()).index(event_name))),font, 1, color, 2, cv2.LINE_AA)
            # Display face values
            for i, (blendshape, value) in enumerate(face_values.items()):
                blendshape_text = f'{blendshape}: {round(value,5)}'
                cv2.putText(image, blendshape_text, (10, 30 * (i + 1 + len(face_status) + 1)),font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)


        # Hands
        """ ===========================
                    RIGHT HAND
        =============================== """ 
        if results.right_hand_landmarks:
            # Gets Hand coords
            right_hand_landmarks = results.right_hand_landmarks.landmark
            x_min = int(min(right_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*(1-margin_hand.left))
            x_max = int(max(right_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*(1+margin_hand.right))
            y_min = int(min(right_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*(1-margin_hand.bottom))
            y_max = int(max(right_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*(1+margin_hand.top))
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
            Hand_results = HandRecognizer.recognize(mp_image)
            if Hand_results.gestures :
                hands_gesture['RIGHT'] = Hand_results.gestures[0][0].category_name if (Hand_results.gestures[0][0].category_name != '' and Hand_results.gestures[0][0].category_name != 'none')  else None
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
            x_min = int(min(left_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*(1-margin_hand.left))
            x_max = int(max(left_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*(1+margin_hand.right))
            y_min = int(min(left_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*(1-margin_hand.bottom))
            y_max = int(max(left_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*(1+margin_hand.top))
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
            Hand_results = HandRecognizer.recognize(mp_image)
            if Hand_results.gestures :
                hands_gesture['LEFT'] = Hand_results.gestures[0][0].category_name if (Hand_results.gestures[0][0].category_name != '' and Hand_results.gestures[0][0].category_name != 'none')  else None
            else:
                hands_gesture['LEFT'] = None
        else :
            hands_gesture['LEFT'] = None

        # Display hand results   
        if face_values:
            for event_name, event_state in hands_gesture.items():
                text = f'{event_name}: {event_state}'
                color = (0, 255, 0)
                y_position = 30 * (1+ 1 + len(face_status) + 1 + len(face_values) + list(hands_gesture.keys()).index(event_name))
                cv2.putText(image, text, (10, y_position), font, 1, color, 2, cv2.LINE_AA)
                # cv2.putText(image, text, (10, 30 * (1 + list(hands_gesture.keys()).index(event_name))),font, 1, color, 2, cv2.LINE_AA)
        

        """
        ==========================================================================================================
        DISPLAY
        ==========================================================================================================
        """  
        # Recolor Feed into BGR for Display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image,1)

        # Display frame cropped face
        # face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)
        # face_frame = cv2.flip(face_frame,1)
        # cv2.imshow('FACE', face_frame)

        # Display hand frame
        # if results.right_hand_landmarks:
        #     right_hand_frame = cv2.cvtColor(right_hand_frame, cv2.COLOR_RGB2BGR)
        #     right_hand_frame = cv2.flip(right_hand_frame,1)
        #     cv2.imshow('RIGHT HAND', right_hand_frame)
        
        # if results.left_hand_landmarks:
        #     left_hand_frame = cv2.cvtColor(left_hand_frame, cv2.COLOR_RGB2BGR)
        #     left_hand_frame = cv2.flip(left_hand_frame,1)
        #     cv2.imshow('LEFT HAND', left_hand_frame)


        # Display hand results    
        for event_name, event_state in hands_gesture.items():
            text = f'{event_name}: {event_state}'
            color = (0, 255, 0)
            y_position = 30 * (1+ 1 + len(face_status) + 1 + len(face_values) + list(hands_gesture.keys()).index(event_name))
            cv2.putText(image, text, (10, y_position), font, 1, color, 2, cv2.LINE_AA)
            # cv2.putText(image, text, (10, 30 * (1 + list(hands_gesture.keys()).index(event_name))),font, 1, color, 2, cv2.LINE_AA)

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
