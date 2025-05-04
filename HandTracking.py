import mediapipe as mp
import cv2
import math


class HandDetector:
    def __init__(self, mode=False, num_hands=2, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.num_hands = num_hands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpHand = mp.solutions.hands
        self.hand = self.mpHand.Hands(
            static_image_mode=mode,
            max_num_hands=num_hands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def detectHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hand.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, landmarks, self.mpHand.HAND_CONNECTIONS)
        return img

    def find_landmarks(self, img):
        landmarks_list = []
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                hand_points = []
                for id, landmark in enumerate(hand_landmarks.landmark):
                    hand_points.append({'id': id, 'x': landmark.x, 'y': landmark.y})
                landmarks_list.append(hand_points)
        return landmarks_list  # Returns list of hands, each with list of landmarks

    def calculateDistance(self, id1, id2, landmarks):
        if not landmarks or not landmarks[0]:  # Check if landmarks exist and first hand is present
            return None

        hand = landmarks[0]  # Use first detected hand
        point1 = next((item for item in hand if item['id'] == id1), None)
        point2 = next((item for item in hand if item['id'] == id2), None)

        if point1 and point2:
            dx = point1['x'] - point2['x']
            dy = point1['y'] - point2['y']
            distance = math.sqrt(dx ** 2 + dy ** 2)
            return distance
        return None
