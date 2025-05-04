import cv2
import mediapipe as mp
import math
import time
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def hand_gesture_rectangle_drawing():
    cap = cv2.VideoCapture(0)
    points = []
    last_point_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        index_points = {}
        h, w = frame.shape[:2]

        if results.multi_hand_landmarks and len(results.multi_handedness) == len(results.multi_hand_landmarks):
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                label = results.multi_handedness[idx].classification[0].label

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                index_points[label] = (x, y)

                cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)

        # If both hands are visible and index tips are close, consider it a valid point
        if 'Left' in index_points and 'Right' in index_points:
            lx, ly = index_points['Left']
            rx, ry = index_points['Right']
            dist = math.hypot(lx - rx, ly - ry)

            if dist < 10 and len(points) < 3:  # Pixel distance threshold
                current_time = time.time()
                if current_time - last_point_time >= 1.5:
                    point = ((lx + rx) // 2, (ly + ry) // 2)
                    if len(points) == 0 or points[-1] != point:
                        points.append(point)
                        last_point_time = current_time

        # Draw saved points
        for i, point in enumerate(points):
            cv2.circle(frame, point, 6, (0, 255, 0), -1)
            cv2.putText(frame, f'{i+1}: {point}', (point[0] + 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if len(points) == 3:
            # Use first point as center
            center = np.array(points[0], dtype=np.int32)
            p1 = np.array(points[1], dtype=np.int32)
            p2 = np.array(points[2], dtype=np.int32)

            # Normalize both vectors and scale to equal length
            v1 = p1 - center
            v2 = p2 - center
            length = 150  # Fixed length for both arms of Y
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)

            p1_ext = tuple((center + v1_norm * length).astype(int))
            p2_ext = tuple((center + v2_norm * length).astype(int))

            # Draw the two main arms of Y
            cv2.line(frame, tuple(center), p1_ext, (0, 0, 255), 2)
            cv2.line(frame, tuple(center), p2_ext, (0, 0, 255), 2)

            # Compute third arm (Y tail) as vector bisector downward
            bisector = -((v1_norm + v2_norm) / 2)
            bisector /= np.linalg.norm(bisector)
            tail_point = tuple((center + bisector * length).astype(int))

            # Draw tail of Y
            cv2.line(frame, tuple(center), tail_point, (0, 255, 255), 2)

            # Calculate angle between v1 and v2
            dot_product = np.dot(v1_norm, v2_norm)
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            # Display angle
            cv2.putText(frame, f'Angle: {angle_deg:.2f} deg',
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Optional: label the arms
            cv2.putText(frame, 'Y shape formed!', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Y Shape Drawer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

hand_gesture_rectangle_drawing()
