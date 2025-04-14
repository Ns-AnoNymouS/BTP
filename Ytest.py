import cv2
import mediapipe as mp
import math

# Constants
PIXELS_PER_CM = 10  # Placeholder: adjust according to calibration

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

REACHING_TOE = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
STATIC_TOE = mp_pose.PoseLandmark.LEFT_FOOT_INDEX
REFERENCE_HIP = mp_pose.PoseLandmark.RIGHT_HIP

cap = cv2.VideoCapture('./videos/sta1.mp4')

initial_position = None
frame_count = 0
reach_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Crop: remove ~1cm (10 pixels) from top, left, right
    frame = frame[10:h, 10:w-10]

    h, w, _ = frame.shape  # Update dimensions after cropping
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        moving_toe = landmarks[REACHING_TOE]
        static_toe = landmarks[STATIC_TOE]
        hip = landmarks[REFERENCE_HIP]

        x = int(moving_toe.x * w)
        y = int(moving_toe.y * h)
        static_x = int(static_toe.x * w)
        static_y = int(static_toe.y * h)
        hip_x = int(hip.x * w)

        if initial_position is None and frame_count > 10:
            initial_position = (x, y)
            print(f"Initial Position: {initial_position}")
        elif initial_position:
            dx = x - initial_position[0]
            dy = y - initial_position[1]

            direction = ""
            distance_cm = 0

            if dx > 30 and abs(dy) < 40:
                direction = "Anterior Reach"
                distance_cm = abs(x - static_x) / PIXELS_PER_CM  # Horizontal
            elif dx < -30:
                pixel_distance = math.hypot(x - static_x, y - static_y)
                distance_cm = pixel_distance / PIXELS_PER_CM     # Diagonal
                if y < static_y + 8:
                    direction = "Posterolateral Reach"
                else:
                    direction = "Posteromedial Reach"

            if direction:
                text = f"{direction}: {distance_cm:.1f} cm"
                cv2.putText(frame, text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if not reach_detected:
                    print(f"Reach Detected: {direction}, Distance: {distance_cm:.1f} cm")
                    reach_detected = True

        # Draw points
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)  # Reaching toe
        cv2.circle(frame, (static_x, static_y), 6, (0, 255, 255), -1)  # Static toe
        cv2.circle(frame, (hip_x, int(hip.y * h)), 6, (0, 0, 255), -1)  # Hip
        
    # === Zoom into bottom part and strip sides ===
    zoom_factor = 1 # Change this for more or less zoom
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    x1 = int((w - new_w) / 2)
    y1 = h - new_h
    x2 = x1 + new_w
    y2 = h
    
    # Crop and resize
    zoomed_frame = frame[y1:y2, x1:x2]
    frame = cv2.resize(zoomed_frame, (w, h))

    cv2.imshow("Y Balance Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    

    frame_count += 1

cap.release()
cv2.destroyAllWindows()