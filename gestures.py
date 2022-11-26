import cv2
import mediapipe as mp
import time
import threading


class Gestures(threading.Thread):
    def __init__(self):
        super().__init__()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.n_poses = 10
        self.recognition_time = time.time()
        self.previous_poses = []

        self.cap = cv2.VideoCapture(0)

    def run(self):
        with self.mp_hands.Hands(
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.results = hands.process(image).multi_hand_landmarks

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if self.results:
                    # print(self.get_gesture())
                    if len(self.previous_poses) == self.n_poses:
                        self.previous_poses.pop(0)
                    self.previous_poses.append(self.results[0].landmark[8])
                    for hand_landmarks in self.results:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.cap.release()

    def get_gesture(self):
        if self.results:
            if self.check_if_only_index_finger() and len(self.previous_poses) == self.n_poses and (
                    time.time() - self.recognition_time > 2):
                index_finger = self.results[0].landmark[8]
                if index_finger.x - self.previous_poses[0].x > 0.2:
                    self.recognition_time = time.time()
                    return "left"
                if self.previous_poses[0].x - index_finger.x > 0.2:
                    self.recognition_time = time.time()
                    return "right"
                if index_finger.y - self.previous_poses[0].y < -0.2:
                    self.recognition_time = time.time()
                    return "up"
                if self.previous_poses[0].y - index_finger.y < -0.2:
                    self.recognition_time = time.time()
                    return "down"
        return False

    def check_if_only_index_finger(self):
        finger_indexes = [8, 12, 16, 20]
        fingers_opened = []
        for finger_index in finger_indexes:
            finger_end = self.results[0].landmark[finger_index]
            finger_begins = self.results[0].landmark[finger_index - 2]
            if finger_end.y > finger_begins.y:
                fingers_opened.append(0)
            else:
                fingers_opened.append(1)
        return fingers_opened == [1, 0, 0, 0]
