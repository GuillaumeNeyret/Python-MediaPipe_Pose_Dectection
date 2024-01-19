import asyncio
import websockets
import cv2
import json
from typing import Dict

import mediapipe as mp
from parameters import *
import time, cv2
from mediapipe.tasks.python.vision.face_landmarker import Blendshapes
from typing import List, Dict
from margins import Margins
from mediapipe.python.solutions.hands import HandLandmark

async def envoyer_info_websocket(info: Dict):
    uri = "ws://localhost:8765"  # Adresse du serveur WebSocket
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(info))

async def main():


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
        # if blend_values['EYE_BLINK_LEFT']-blend_values['EYE_BLINK_RIGHT']>event_triggers.BLINK_DIFF and blend_values['EYE_BLINK_LEFT']>event_triggers.BLINK:
        if blend_values['EYE_BLINK_LEFT']>event_triggers.BLINK and blend_values['EYE_BLINK_RIGHT']<event_triggers.NO_BLINK and blend_values['EYE_BLINK_LEFT']-blend_values['EYE_BLINK_RIGHT']>event_triggers.BLINK_DIFF:
            return True
        return False

    # Returns True if Smile
    def smile(blend_values:Dict[str,float])-> bool:
        if blend_values['MOUTH_SMILE_LEFT']>event_triggers.SMILE and blend_values['MOUTH_SMILE_LEFT']>event_triggers.SMILE:
            return True
        return False

    # Returns True if surprise
    def surprise(blend_values:Dict[str,float])-> bool:
        # Surprise if Brows Up and No Blink and Jaw Open
        if (blend_values["BROW_OUTER_UP_LEFT"]>event_triggers.BROW_UP and blend_values["BROW_OUTER_UP_RIGHT"]>event_triggers.BROW_UP and 
            blend_values['EYE_BLINK_LEFT']<event_triggers.NO_BLINK and blend_values['EYE_BLINK_RIGHT']<event_triggers.NO_BLINK and
            blend_values['JAW_OPEN']>event_triggers.OPEN_MOUTH):
            return True
        return False

    # Returns True if angry
    def angry(blend_values:Dict[str,float])-> bool:
        # Trigger if Brow Down and Mouth Closed
        if blend_values['BROW_DOWN_LEFT']>event_triggers.BROW_DOWN and blend_values['BROW_DOWN_RIGHT']>event_triggers.BROW_DOWN and blend_values['JAW_OPEN']<event_triggers.NO_OPEN_MOUTH :
            return True
        return False

    # Return face expression as a dict
    def event_faces(blend_values: Dict[str, float]) -> Dict[str, bool]:
        return {
            'Smile': smile(blend_values),
            'Kiss': kiss(blend_values),
            'Left_blink': left_blink(blend_values),
            'Surprise': surprise(blend_values),
            'Angry': angry(blend_values) if not smile(blend_values) and not kiss(blend_values) else False
        }

    def triggering_status(face_status,hands_gesture):
        left_hand, right_hand = hands_gesture['LEFT'] , hands_gesture['RIGHT']

        res = {
            'Kiss': face_status['Kiss'],
            'Smile':face_status['Smile'],
            'Wink': face_status['Left_blink'],
            'Surprise': face_status['Surprise'],
            'Zen':True if (left_hand, right_hand) == ('zen','zen') else None,
            'Pulp_Fiction':None,
            'Okay': True if left_hand == 'okay' else None,
            'Left_Thumb':True if left_hand == 'thumb_up' else None,
            'Angry': face_status['Angry']
        }

        return res

    def triggering_status_init():
        res = {
            'Kiss': None,
            'Smile':None,
            'Wink': None,
            'Surprise': None,
            'Zen':None,
            'Pulp_Fiction':None,
            'Okay': None,
            'Left_Thumb':None,
            'Angry': None
        }

        return res


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

    trigger_status = triggering_status_init()

    face_status={
        'Smile': None,
        'Kiss': None,
        'Left_blink': None,
        'Surprise': None,
        'Angry': None
        }

    hands_gesture = {
        "LEFT":None,
        "RIGHT": None
    }

    face_values = {i : None for i in blend_list}


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

            # # Crop to fit screen
            # cropped = frame[center[0] - (crop_dim[0]) // 2: center[0] + (crop_dim[0]) // 2,
            #                 center[1] - (crop_dim[1]) // 2: center[1] + (crop_dim[1]) // 2]
            # cv2.imshow('CROP', cropped)

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
                x_min = int(max(0,min(head_landmarks, key=lambda x: x.x).x * image.shape[1]))
                x_max = int(max(0,max(head_landmarks, key=lambda x: x.x).x * image.shape[1]))
                y_min = int(max(0,min(head_landmarks, key=lambda y: y.y).y * image.shape[0]))
                y_max = int(max(0,max(head_landmarks, key=lambda y: y.y).y * image.shape[0]))
                # Extract ROI
                ROI = image[y_min:y_max, x_min:x_max]
                face_frame = ROI.copy()

                # FACE BELNDSHAPES RECOGNIZER
                mp_Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_frame)
                face_result = FaceRecognizer.detect_for_video(mp_Image,timestamp)

                if face_result != None and face_result.face_blendshapes:
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
                

            # Hands
            """ ===========================
                        RIGHT HAND
            =============================== """ 
            if results.right_hand_landmarks:
                # Gets Hand coords
                right_hand_landmarks = results.right_hand_landmarks.landmark
                x_min = int(max(0,min(right_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*(1-margin_hand.left)))
                x_max = int(max(0,max(right_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*(1+margin_hand.right)))
                y_min = int(max(0,min(right_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*(1-margin_hand.bottom)))
                y_max = int(max(0,max(right_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*(1+margin_hand.top)))
                # Extract ROI
                ROI = image[y_min:y_max, x_min:x_max]
                right_hand_frame = ROI.copy()
                # print(f"x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")
                # print('Right Hand shapes :', right_hand_frame.shape)            

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
                x_min = int(max(0,min(left_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*(1-margin_hand.left)))
                x_max = int(max(0,max(left_hand_landmarks, key=lambda x: x.x).x * image.shape[1]*(1+margin_hand.right)))
                y_min = int(max(0,min(left_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*(1-margin_hand.bottom)))
                y_max = int(max(0,max(left_hand_landmarks, key=lambda y: y.y).y * image.shape[0]*(1+margin_hand.top)))
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
            
            trigger_status = triggering_status(face_status,hands_gesture)

            # Pulp Fiction case
            if hands_gesture['LEFT'] == 'pulp_fiction' or hands_gesture['RIGHT'] == 'pulp_fiction':
                if hands_gesture['LEFT'] == 'pulp_fiction':
                    pass
                if hands_gesture['RIGHT'] == 'pulp_fiction' and results.face_landmarks:
                    # Get max, min coords from the 2 fingers
                    INDEX_ID = [HandLandmark.INDEX_FINGER_MCP,HandLandmark.INDEX_FINGER_PIP,HandLandmark.INDEX_FINGER_DIP,HandLandmark.INDEX_FINGER_TIP]
                    MIDDLE_ID = [HandLandmark.MIDDLE_FINGER_MCP,HandLandmark.MIDDLE_FINGER_PIP,HandLandmark.MIDDLE_FINGER_DIP,HandLandmark.MIDDLE_FINGER_TIP]

                    x_index_values = [results.right_hand_landmarks.landmark[id].x for id in INDEX_ID]
                    y_index_values = [results.right_hand_landmarks.landmark[id].y for id in INDEX_ID]
                    x_middle_values = [results.right_hand_landmarks.landmark[id].x for id in MIDDLE_ID]
                    y_middle_values = [results.right_hand_landmarks.landmark[id].y for id in MIDDLE_ID]

                    x_finger_min = min(min(x_index_values),min(x_middle_values))
                    x_finger_max = max(max(x_index_values),max(x_middle_values))
                    y_finger_min = min(min(y_index_values),min(y_middle_values))
                    y_finger_max = max(max(y_index_values),max(y_middle_values))

                    right_eye_x = (results.face_landmarks.landmark[33].x+results.face_landmarks.landmark[133].x)/2
                    right_eye_y = (results.face_landmarks.landmark[159].y+results.face_landmarks.landmark[145].y)/2
                    left_eye_x = (results.face_landmarks.landmark[362].x+results.face_landmarks.landmark[398].x)/2
                    left_eye_y = (results.face_landmarks.landmark[159].y+results.face_landmarks.landmark[145].y)/2

                    epsilon = 0.1
                    # RIGHT EYE CLOSED TO PULP FUCTION SIGN
                    if x_finger_min*(1-epsilon)<right_eye_x<x_finger_max*(1+epsilon) and y_finger_min*(1-epsilon)<right_eye_y<y_finger_max*(1+epsilon):
                        trigger_status['Pulp_Fiction']=True


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

            # Display face results  
            for event_name, event_state in face_status.items():
                text = f'{event_name}: {event_state}'
                color = (0, 255, 0) if event_state else (0, 0, 255)
                cv2.putText(image, text, (10, 30 * (1 + list(face_status.keys()).index(event_name))),font, 1, color, 2, cv2.LINE_AA)
            # Display face values
            for i, (blendshape, value) in enumerate(face_values.items()):
                blendshape_text = f'{blendshape}: {round(value,5)}' if value!=None else f'{blendshape}: {value}'
                cv2.putText(image, blendshape_text, (10, 30 * (i + 1 + len(face_status) + 1)),font, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

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

            # Display triggers
            k = 0
            for i, (trigger, status) in enumerate(trigger_status.items()):
                if status:
                    text = f'{trigger} Trigger'
                    cv2.putText(image, text, (10, 800+50 * (k + 1)), font, 2, (255,0,0), 4, cv2.LINE_AA)
                    k+=1

            # Display the resulting frame
            # scale = 60
            # width = int(image.shape[1]*scale/100)
            # height = int(image.shape[0]*scale/100)
            # dim = (width,height)
            # resized = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
            # cv2.imshow('MP FACE', resized)
            cv2.imshow('MP FACE', image)

            # Press 'q' or 'Esc" to exit
            if cv2.waitKey(5) & 0xFF in [ord('q'), 27]:
                break

    cam.release()
    cv2.destroyAllWindows()


    # À l'endroit approprié dans votre boucle principale, appelez la fonction pour envoyer les informations WebSocket
    await envoyer_info_websocket({'face_status': face_status, 'hands_gesture': hands_gesture, 'trigger_status': trigger_status})

if __name__ == "__main__":
    # Lancer le serveur WebSocket en parallèle avec votre application
    start_server = websockets.serve(envoyer_info_websocket, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)

    # Lancer votre application principale
    asyncio.get_event_loop().run_until_complete(main())