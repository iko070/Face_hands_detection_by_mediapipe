import cv2
import mediapipe as mp
import numpy as np
import time
import math

class Autodetec:
    def __init__(self):
        # 初始化medialpipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands

    # 主函数
    def recognize(self):
        # 计算刷新率
        fpsTime = time.time()

        # OpenCV读取视频流
        cap = cv2.VideoCapture(0)
        # 视频分辨率
        resize_w = 640
        resize_h = 480

        # 画面显示初始化参数
        rect_height = 0
        rect_percent_text = 0

        drawing_spec = self.mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)

        with self.mp_face_mesh.FaceMesh(max_num_faces = 2, 
                                        refine_landmarks = True, 
                                        min_detection_confidence = 0.5,
                                        min_tracking_confidence = 0.5) as face_mesh, \
             self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:
            
            while cap.isOpened():
                success, image = cap.read()
                image = cv2.resize(image, (resize_w, resize_h))

                if not success:
                    print("empty")
                    continue
                
                image.flags.writeable = False
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results_face = face_mesh.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                                                        image=image,
                                                        landmark_list=face_landmarks,
                                                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                                        landmark_drawing_spec=None,
                                                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        self.mp_drawing.draw_landmarks(
                                                        image=image,
                                                        landmark_list=face_landmarks,
                                                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                                        landmark_drawing_spec=None,
                                                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                        self.mp_drawing.draw_landmarks(
                                                        image=image,
                                                        landmark_list=face_landmarks,
                                                        connections=self.mp_face_mesh.FACEMESH_IRISES,
                                                        landmark_drawing_spec=None,
                                                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                
                results_hands = hands.process(image)
                # Determine if there are palms
                if results_hands.multi_hand_landmarks:
                    # Traversing each palm
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        # Mark the finger on the screen
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                        # Parse the fingers and store the coordinates of each finger
                        landmark_list = []
                        for landmark_id, finger_axis in enumerate(
                                hand_landmarks.landmark):
                            landmark_list.append([
                                landmark_id, finger_axis.x, finger_axis.y,
                                finger_axis.z
                            ])
                        if landmark_list:
                            # Get thumb-tip coordinates
                            thumb_finger_tip = landmark_list[4]
                            thumb_finger_tip_x = math.ceil(thumb_finger_tip[1] * resize_w)
                            thumb_finger_tip_y = math.ceil(thumb_finger_tip[2] * resize_h)
                            # Get the index fingertip coordinates
                            index_finger_tip = landmark_list[8]
                            index_finger_tip_x = math.ceil(index_finger_tip[1] * resize_w)
                            index_finger_tip_y = math.ceil(index_finger_tip[2] * resize_h)
                            # Middle Point
                            # finger_middle_point = (thumb_finger_tip_x + index_finger_tip_x) // 2, (
                            #         thumb_finger_tip_y + index_finger_tip_y) // 2
                            # print(thumb_finger_tip_x)
                            thumb_finger_point = (thumb_finger_tip_x, thumb_finger_tip_y)
                            index_finger_point = (index_finger_tip_x, index_finger_tip_y)
                            # Draw fingertip 2 dots
                            image = cv2.circle(image, thumb_finger_point, 10, (255, 0, 255), -1)
                            image = cv2.circle(image, index_finger_point, 10, (255, 0, 255), -1)
                            #image = cv2.circle(image, finger_middle_point, 10, (255, 0, 255), -1)
                            # Draw a 2-point line
                            #image = cv2.line(image, thumb_finger_point, index_finger_point, (255, 0, 255), 5)
                            # # Pythagorean theorem calculates the length
                            # line_len = math.hypot((index_finger_tip_x - thumb_finger_tip_x),
                            #                       (index_finger_tip_y - thumb_finger_tip_y))


                # Display refresh rate FPS
                cTime = time.time()
                fps_text = 1 / (cTime - fpsTime)
                fpsTime = cTime
                cv2.putText(image, "FPS: " + str(int(fps_text)), (10, 70),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                
                # Display screen
                cv2.imshow('Autodetec Ver. I feel good, no need for further improvement', image)
                if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('Autodetec Ver. I feel good, no need for further improvement', cv2.WND_PROP_VISIBLE) < 1:
                    break
            cap.release()   

detec = Autodetec()
detec.recognize()
