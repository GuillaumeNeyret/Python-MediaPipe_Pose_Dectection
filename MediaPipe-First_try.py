import mediapipe as mp
from parameters import *
import time, cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic                 # Load Holistic module

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set Camera Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_cam_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_cam_height)


# # Set Window Size
cv2.namedWindow('Holistic of test image', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Holistic of test image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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


# Initializes holistic model
with mp_holistic.Holistic(**settings) as holistic:      # Create holistic object
    if mode == mode_img:
        # IMG PROCESS FOR TEST
        # Make Detections
        results = holistic.process(image)

        # Draw landmarks
        # Face
        # FACEMESH_TESSELATION or FACEMESH_CONTOURS
        mp_drawing.draw_landmarks(image= image,
                                  landmark_list= results.face_landmarks,
                                  connections= mp_holistic.FACEMESH_CONTOURS,
                                  landmark_drawing_spec = mp_drawing.DrawingSpec(**draw_face_landmark),
                                  connection_drawing_spec = mp_drawing.DrawingSpec(**draw_face_connection)
                                  )
        # Hands
        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=results.right_hand_landmarks,
                                  connections=mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_landmark),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_connection)
                                  )
        mp_drawing.draw_landmarks(image=image,
                                  landmark_list=results.left_hand_landmarks,
                                  connections=mp_holistic.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_landmark),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_connection)
                                  )

        # Body
        if results.pose_landmarks:  # if it finds the points
            for id,landmrk in enumerate(results.pose_landmarks.landmark):
                # print(id,landmrk)
                if id in excluded_index_pose :
                    landmrk.visibility = 0
            # print('H\n','Type:',type(h),'\n Values:\n',h ,'\n Values x:\n',h.visibility)

            mp_drawing.draw_landmarks(image=image,
                                      landmark_list=results.pose_landmarks,
                                      connections=CUSTOM_BODY_CONNECTION,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_body_landmark),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(**draw_body_connection)
                                      )


        if image is not None:
            cv2.imshow('Holistic of test image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif mode == mode_cam :
        # CAM PROCESS
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                print("Can't receive frame ...")
                err += 1
                if err == 3:                                 # 3 Consecutive unreadable frame stop the process
                    break
                continue
            err = 0

            image = frame
            # Recolor Feed into RGB for MP
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)

            # Draw landmarks
            # Face
            # FACEMESH_CONTOURS or FACEMESH_TESSELATION
            mp_drawing.draw_landmarks(image=image,
                                      landmark_list=results.face_landmarks,
                                      connections= mp_holistic.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_face_landmark),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(**draw_face_connection)
                                      )
            # Hands
            mp_drawing.draw_landmarks(image=image,
                                      landmark_list=results.right_hand_landmarks,
                                      connections=mp_holistic.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_landmark),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_connection)
                                      )
            mp_drawing.draw_landmarks(image=image,
                                      landmark_list=results.left_hand_landmarks,
                                      connections=mp_holistic.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_landmark),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(**draw_hand_connection)
                                      )

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

            # Recolor Feed into BGR for Display
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # FPS Display
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            cv2.putText(image, str(fps), (500, 500), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


            # Display the resulting frame
            cv2.imshow('Holistic Pose Detection', image)

            # Press 'q' or 'Esc" to exit
            if cv2.waitKey(5) & 0xFF in [ord('q'), 27]:
                break

cam.release()
cv2.destroyAllWindows()
