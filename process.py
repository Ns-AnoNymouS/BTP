import os
import cv2
import mediapipe as mp


# Initialize Mediapipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils


def get_stability(input_path, output_path) -> str:
    """
    Analyze video for stability by plotting COG and BOS and detecting instability.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video file.

    Returns:
        str: Path to the re-encoded output video file.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Error: Cannot open video at {input_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30  # Use default if FPS is 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use lowercase codec
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    try:
        while cap.isOpened():
            print("Processiong frame: {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )

            out.write(frame)

    finally:
        cap.release()
        out.release()

    return output_path


# Example usage
output_file = get_stability("videos/tharun.mp4", "videos/dance_stability.mp4")
print(f"Processed video saved at: {output_file}")
