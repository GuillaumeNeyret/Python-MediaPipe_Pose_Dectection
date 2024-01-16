import mediapipe as mp
from parameters import *
import time, cv2, math

RIGHT_EYE = {'right':33,'left':133,'top':159,'bottom':145}
LEFT_EYE = {'right':362,'left':263,'top':386,'bottom':374}

def landmarksDetection(img_height, img_width, results):
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    return mesh_coord

def distance(p1,p2):
    x1,y1 = p1
    x2, y2 = p2
    distance= math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
def blink_ratio(landmarks):
    rh_right, rh_left, rv_top, rv_bottom = landmarks[RIGHT_EYE['right']],landmarks[RIGHT_EYE['left']], landmarks[RIGHT_EYE['top']], landmarks[RIGHT_EYE['bottom']]
    lh_right, lh_left, lv_top, lv_bottom = landmarks[LEFT_EYE['right']], landmarks[LEFT_EYE['left']], landmarks[LEFT_EYE['top']], landmarks[LEFT_EYE['bottom']]

    rhDistance = distance(rh_right,rh_left)
    rvDistance = distance(rv_top, rv_bottom)
    lhDistance = distance(lh_right, lh_left)
    lvDistance = distance(lv_top, lv_bottom)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance
    ratio = (reRatio + leRatio) / 2
    return (reRatio,leRatio,ratio)



map_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Set Camera Resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, res_cam_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res_cam_height)

# Set Window Size
cv2.namedWindow('MP TEST', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('MP TEST', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


with map_face_mesh.FaceMesh(**settings_face_mesh) as face_mesh:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame ...")
            break


        # ZOOM IN NEEDED ???
        image = cv2.resize(frame,dsize=(res_cam_width,res_cam_height),fx=3,fy=3, interpolation=cv2.INTER_CUBIC)
        image_height, image_width = image.shape[:2]
        # Recolor Feed into RGB for MP
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = face_mesh.process(image)

        # Draw Face landmarks
        # FACEMESH_CONTOURS or FACEMESH_TESSELATION
        if results.multi_face_landmarks:
            # Draw Face landmarks
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image=image,
                                          landmark_list= face_landmarks,
                                          connections=map_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing.DrawingSpec(**draw_face_connection)
                                          )

            # Mesh coordinates
            print(image_height, "//", image_width)
            mesh_coords = landmarksDetection(img_height=image_height, img_width=image_width , results=results)
            rratio,lratio,ratio = blink_ratio(landmarks=mesh_coords)


        # Recolor Feed into BGR for Display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image = cv2.flip(image, 1)
        if results.multi_face_landmarks:
            blin = abs(rratio-lratio)*4
            cv2.putText(image, f'LEFT RATIO : {round(lratio,2)} // RIGHT RATIO : {round(rratio,2)} // BLINK : {round(blin,2)} // RATIO : {round(ratio,2)}', (500, 500), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('MP TEST', image)

        # Press 'q' or 'Esc" to exit
        if cv2.waitKey(5) & 0xFF in [ord('q'), 27]:
            break

cam.release()
cv2.destroyAllWindows()
