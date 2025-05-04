import cv2
import mediapipe as mp
import math
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate distance between two landmarks
def calculate_distance(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

def capture_intersection_points():
    cap = cv2.VideoCapture(0)
    points = []
    last_point_time = 0
    delay = 1.5  # seconds between point confirmations

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_data = {}

        if results.multi_hand_landmarks and len(results.multi_handedness) == len(results.multi_hand_landmarks):
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
                index_coords = (int(index_tip.x * w), int(index_tip.y * h))

                cv2.circle(frame, thumb_coords, 6, (255, 255, 0), -1)
                cv2.circle(frame, index_coords, 6, (255, 0, 255), -1)

                hand_data[label] = {
                    "thumb": thumb_tip,
                    "index": index_tip,
                    "thumb_coords": thumb_coords,
                    "index_coords": index_coords
                }

        # Condition: both hands are visible and making the matching gesture
        if 'Left' in hand_data and 'Right' in hand_data and len(points) < 2:
            l = hand_data['Left']
            r = hand_data['Right']

            # Thumb intersection (tips of thumb from both hands)
            thumb_dist = calculate_distance(l['thumb'], r['thumb'])
            # Index intersection (tips of index from both hands)
            index_dist = calculate_distance(l['index'], r['index'])

            if thumb_dist < 0.05 and index_dist < 0.05:
                current_time = time.time()
                if current_time - last_point_time >= delay:
                    # Point for Thumb intersection
                    thumb_point = (int((l['thumb_coords'][0] + r['thumb_coords'][0]) / 2),
                                   int((l['thumb_coords'][1] + r['thumb_coords'][1]) / 2))
                    points.append(thumb_point)

                    # Point for Index intersection
                    index_point = (int((l['index_coords'][0] + r['index_coords'][0]) / 2),
                                   int((l['index_coords'][1] + r['index_coords'][1]) / 2))
                    points.append(index_point)

                    last_point_time = current_time

        # Draw all recorded points
        for i, pt in enumerate(points):
            cv2.circle(frame, pt, 8, (0, 255, 0), -1)
            cv2.putText(frame, f'Point {i+1}', (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw lines between points if both points are recorded
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (0, 255, 255), 2)

        cv2.imshow("Thumb and Index Intersection Points", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_intersection_points()
