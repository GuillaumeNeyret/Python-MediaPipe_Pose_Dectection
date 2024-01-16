import mediapipe as mp
from parameters import *
import time, cv2
from mediapipe.tasks.python.vision.face_landmarker import Blendshapes
from typing import List, Dict

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic                 # Load Holistic module

BaseOptions = mp.tasks.BaseOptions                  # Gesture Recognizer Loading
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set Camera Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_cam_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_cam_height)


# # Set Window Size
# cv2.namedWindow('MP TEST', cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty('MP TEST', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_frame_time = 0
err = 0

# Get rid of Face and Hand Points of the pose
excluded_index_pose = [0,1,2,3,4,5,6,7,8,9,10,17,18,19,20,21,22]
CUTOFF_THRESHOLD = 10
CUSTOM_BODY_CONNECTION = frozenset([t for t in mp_holistic.POSE_CONNECTIONS if t[0] not in excluded_index_pose and t[1] not in excluded_index_pose ])

# Load Image Test
img_path = 'assets/img/Test3.jpg'
# img_path = 'assets/img/Test6.jpg'
image = cv2.imread(img_path)

gesture = ''

# Create a image segmenter instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    category_name = result.gestures[0][0].category_name
    print(result.gestures)
    print(category_name)
    # global gesture= result.gestures[0][0].category_name
    # return

# Hand Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='assets/models/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

recognizer = GestureRecognizer.create_from_options(options)

# Facial Blendshape model
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='assets/models/face_landmarker_v2_with_blendshapes.task'),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
    )

Face_recognizer = FaceLandmarker.create_from_options(options)

timestamp = 0


# List of values needed from Blendshape
blend_list = [
    "JAW_OPEN",
    "BROW_DOWN_LEFT",
    "BROW_DOWN_RIGHT",
    "MOUTH_CLOSE",
    "MOUTH_PUCKER",
    "EYE_BLINK_LEFT",
    "EYE_BLINK_RIGHT",
    "MOUTH_SMILE_LEFT",
    "MOUTH_SMILE_RIGHT"
    ]

# Transform a blendshape to a dict with only values from the need_value list
def blendshapes_to_dict(face_blendshape):

    return {i : face_blendshape[getattr(Blendshapes, i).value].score for i in blend_list}

# Returns True if Kiss 
def kiss(blend_values:Dict[str,float]):
    if blend_values['MOUTH_PUCKER']> event_triggers.KISS and blend_values["JAW_OPEN"]<event_triggers.NO_OPEN_MOUTH:
        return True
    return False

# Return face expression event as a String
def event_faces(blend_values:Dict[str,float])->str:
    state = None
    if kiss(blend_values):
        state = "Kiss"
    
    return state

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
        # Recolor Feed into RGB for MP
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)

        # Draw landmarks
        # Face
        mp_drawing.draw_landmarks(image=image,
                                    landmark_list=results.face_landmarks,
                                    connections= mp_holistic.FACEMESH_CONTOURS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_face_landmark),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(**draw_face_connection)
                                    )
        # Hands
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image=image,
                                    landmark_list=results.right_hand_landmarks,
                                    connections=mp_holistic.HAND_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_landmark),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_connection)
                                    )
            
            # Hand Gesture Recognizer
            # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # recognizer.recognize_async(mp_image, timestamp)
            
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image=image,
                                    landmark_list=results.left_hand_landmarks,
                                    connections=mp_holistic.HAND_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_landmark),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_connection)
                                    )
            # Hand Gesture Recognizer
            # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # recognizer.recognize_async(mp_image, timestamp)

        # Body
        if results.pose_landmarks:  # if it finds the points
            for id, landmrk in enumerate(results.pose_landmarks.landmark):
                # print(id,landmrk)
                if id in excluded_index_pose:
                    landmrk.visibility = 0
            # print('H\n','Type:',type(h),'\n Values:\n',h ,'\n Values x:\n',h.visibility)

            mp_drawing.draw_landmarks(image=image,
                                        landmark_list=results.pose_landmarks,
                                        connections=CUSTOM_BODY_CONNECTION,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_body_landmark),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(**draw_body_connection)
                                        )
            
        mp_Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        face_result = Face_recognizer.detect_for_video(mp_Image,timestamp)
        if face_result != None:
            face_values = blendshapes_to_dict(face_blendshape=face_result.face_blendshapes[0])
            # print(face_values)
            jawOpen_value = face_result.face_blendshapes[0][Blendshapes.JAW_OPEN.value].score
            face_status = event_faces(face_values)
        else:
            face_status=None



        # Recolor Feed into BGR for Display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = cv2.flip(image,1)


        cv2.putText(image, f"State : {face_status}", (1000, 1000), font, 2, (0, 255, 255), 2, cv2.LINE_AA)

        try:
            cv2.putText(image, f"jawOpen : {round(jawOpen_value,5)}", (200, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        except NameError:
            pass


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
