import cv2
import mediapipe as mp
import math
import time
import numpy as np

# Constants
REAL_WORLD_CM = 8.56  # ATM card width in cm
Y_LENGTH_CM = 120  # Desired real-world Y arm length in cm
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

REACHING_TOE = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
STATIC_TOE = mp_pose.PoseLandmark.LEFT_FOOT_INDEX

initial_position = None
frame_count = 0
reach_detected = False

def calculate_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def point_to_line_distance(pt, line_start, line_end):
    px, py = pt
    x1, y1 = line_start
    x2, y2 = line_end
    if (x1, y1) == (x2, y2):
        return calculate_distance(pt, line_start)
    return abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1) / math.hypot(x2 - x1, y2 - y1)

def classify_direction(toe_point, y_points):
    A, B, C = y_points
    midpoint = ((B[0] + C[0]) // 2, (B[1] + C[1]) // 2)
    distances = {
        "Posteromedial": point_to_line_distance(toe_point, A, B),
        "Posterolateral": point_to_line_distance(toe_point, A, C),
        "Anterior": point_to_line_distance(toe_point, A, midpoint)
    }
    direction = min(distances, key=distances.get)
    final_distances = calculate_distance(toe_point, y_points[0])
    return direction, final_distances

def draw_y_shape(frame, center, p1, p2, pixel_per_cm=None):
    center = np.array(center, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    v1 = p1 - center
    v2 = p2 - center
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    if pixel_per_cm:
        length = Y_LENGTH_CM * pixel_per_cm
    else:
        length = np.linalg.norm(v1)  # use default raw pixel distance

    p1_ext = tuple((center + v1_norm * length).astype(int))
    p2_ext = tuple((center + v2_norm * length).astype(int))

    bisector = -((v1_norm + v2_norm) / 2)
    bisector /= np.linalg.norm(bisector)
    tail_point = tuple((center + bisector * length).astype(int))

    # Check bounds
    all_pts = [tuple(center.astype(int)), p1_ext, p2_ext, tail_point]
    fits_screen = all(
        0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT for (x, y) in all_pts
    )

    if not fits_screen and pixel_per_cm:
        cv2.putText(
            frame,
            "Move the screen farther!",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )
    else:
        cv2.line(frame, tuple(center.astype(int)), p1_ext, (0, 0, 255), 2)
        cv2.line(frame, tuple(center.astype(int)), p2_ext, (0, 0, 255), 2)
        cv2.line(frame, tuple(center.astype(int)), tail_point, (0, 255, 255), 2)

    # Show angle
    angle_rad = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    cv2.putText(
        frame,
        f"Angle: {angle_deg:.2f} deg",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def hand_gesture_y_shape():
    global initial_position, frame_count, reach_detected

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Y Balance Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Y Balance Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    max_reach = {
        "Anterior": 0,
        "Posteromedial": 0,
        "Posterolateral": 0
    }

    points = []
    calibrating_points = []
    distance_points = []
    pixel_per_cm = None
    last_point_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        index_points = {}
        h, w = frame.shape[:2]

        if pixel_per_cm is None:
            if results.multi_hand_landmarks and len(results.multi_handedness) == len(
                results.multi_hand_landmarks
            ):
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    label = results.multi_handedness[idx].classification[0].label

                    index_tip = hand_landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ]
                    x, y = int(index_tip.x * w), int(index_tip.y * h)
                    index_points[label] = (x, y)
                    cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

            if "Left" in index_points and "Right" in index_points:
                lx, ly = index_points["Left"]
                rx, ry = index_points["Right"]
                dist = math.hypot(lx - rx, ly - ry)

                if dist < 10:
                    current_time = time.time()
                    if current_time - last_point_time >= 1.5:
                        point = ((lx + rx) // 2, (ly + ry) // 2)
                        if len(points) < 3:
                            if len(points) == 0 or points[-1] != point:
                                points.append(point)
                                last_point_time = current_time
                        elif len(calibrating_points) < 2:
                            if (
                                len(calibrating_points) == 0
                                or calibrating_points[-1] != point
                            ):
                                calibrating_points.append(point)
                                last_point_time = current_time
                        elif len(distance_points) < 2:
                            if (
                                len(distance_points) == 0
                                or distance_points[-1] != point
                            ):
                                distance_points.append(point)
                                last_point_time = current_time

            # Calibration
            if len(calibrating_points) == 2 and pixel_per_cm is None:
                px_dist = math.dist(calibrating_points[0], calibrating_points[1])
                pixel_per_cm = px_dist / REAL_WORLD_CM
                print(f"[INFO] Calibration done: {pixel_per_cm:.2f} pixels/cm")

            # Draw collected points
            for i, point in enumerate(points):
                cv2.circle(frame, point, 6, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"Y{i+1}: {point}",
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            for i, point in enumerate(calibrating_points):
                cv2.circle(frame, point, 6, (255, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"C{i+1}: {point}",
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

            for i, point in enumerate(distance_points):
                cv2.circle(frame, point, 6, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"D{i+1}: {point}",
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
        else:
            results = pose.process(rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_toe = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                toe_point = (int(left_toe.x * w), int(left_toe.y * h))

                cv2.circle(frame, toe_point, 8, (0, 0, 255), -1)

                if toe_point:
                    direction, px_dist = classify_direction(toe_point, points)
                    dist_cm = round(px_dist / pixel_per_cm, 2)
                    max_reach[direction] = max(max_reach[direction], dist_cm)
                    
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.putText(frame, f"Direction: {direction}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f"Current: {dist_cm:.2f} cm", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 200), 2)
            
        # Always draw Y once 3 points are selected (scaled if calibrated)
        if len(points) == 3:
            center = np.array(points[0], dtype=np.int32)
            p1 = np.array(points[1], dtype=np.int32)
            p2 = np.array(points[2], dtype=np.int32)
            if ((p1[0] < center[0] < p2[0]) or (p2[0] < center[0] < p1[0])) and (
                (p1[1] < center[1] and p2[1] < center[1])
                or (p1[1] > center[1] and p2[1] > center[1])
            ):
                cv2.putText(
                    frame,
                    "You should provide the side view of the Y shape!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
            else:
                draw_y_shape(frame, points[0], points[1], points[2], pixel_per_cm)
        
        cv2.imshow("Y Balance Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Max reach:", max_reach)


hand_gesture_y_shape()
