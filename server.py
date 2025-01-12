import os
import cv2
import subprocess
import mediapipe as mp
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Suppress TensorFlow Lite warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Initialize Flask app
app = Flask(__name__, static_folder="temp", static_url_path="/video")
CORS(app)

# Initialize Mediapipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils


def calculate_cog(landmarks: list, image_width: int, image_height: int) -> tuple | None:
    """
    Calculate the Center of Gravity (COG) based on body keypoints.

    Args:
        landmarks (list): List of body landmarks detected by Mediapipe.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        tuple: (x, y) coordinates of the COG or None if not calculable.
    """
    body_segments = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]
    segment_weights = [10, 10, 20, 20, 15, 15, 5, 5]
    total_weight, weighted_sum_x, weighted_sum_y = 0, 0, 0

    for i, segment in enumerate(body_segments):
        try:
            landmark = landmarks[segment]
            if landmark.visibility > 0.5:
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                weight = segment_weights[i]
                total_weight += weight
                weighted_sum_x += x * weight
                weighted_sum_y += y * weight
        except (IndexError, AttributeError):
            continue

    if total_weight > 0:
        return int(weighted_sum_x / total_weight), int(weighted_sum_y / total_weight)
    return None


def process_video(input_path: str, output_path: str) -> str:
    """
    Process the input video to add COG markers and save the output.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video file.

    Returns:
        str: Path to the re-encoded output video file.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Error: Cannot open video.")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = frame.shape

        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )
            cog_coordinates = calculate_cog(
                results.pose_landmarks.landmark, image_width, image_height
            )
            if cog_coordinates:
                cv2.circle(frame, cog_coordinates, 8, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"COG: {cog_coordinates}",
                    (cog_coordinates[0] + 10, cog_coordinates[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        out.write(frame)

    cap.release()
    out.release()

    # Use ffmpeg to re-encode the video to the desired format (e.g., MP4)
    reencoded_output_path = os.path.join(
        "temp", "formatted_" + os.path.basename(input_path)
    )
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        output_path,
        "-vcodec",
        "libx264",
        "-acodec",
        "aac",
        "-strict",
        "experimental",
        reencoded_output_path,
    ]
    subprocess.run(ffmpeg_command, check=True)

    # Clean up temporary files
    if os.path.exists(output_path):
        os.remove(output_path)

    return reencoded_output_path


def get_stability(input_path, output_path) -> str:
    """
    Analyze video for stability by plotting COG and BOS and detecting instability.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video file.

    Returns:
        str: Path to the re-encoded output video file.
    """
    fall_count = 0
    is_stable = False
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Error: Cannot open video.")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter.fourcc(*"MP4v")
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = frame.shape

        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
            )
            cog_coordinates = calculate_cog(
                results.pose_landmarks.landmark, image_width, image_height
            )
            if cog_coordinates:
                cv2.circle(frame, cog_coordinates, 8, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"COG: {cog_coordinates}",
                    (cog_coordinates[0] + 10, cog_coordinates[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            # Extract keypoints
            keypoints = results.pose_landmarks.landmark

            # Define BOS (e.g., foot region - here using LEFT_FOOT_INDEX)
            base_x = keypoints[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].x
            base_y = keypoints[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].y

            # Stability condition
            stable = (
                abs(cog_coordinates[0] / frame.shape[1] - base_x) < 0.1
            )  # Example threshold

            cv2.circle(
                frame,
                (int(base_x * frame.shape[1]), int(base_y * frame.shape[0])),
                5,
                (255, 255, 0),
                -1,
            )
            cv2.putText(
                frame,
                f"Falls: {fall_count}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Notify user about instability
            if not stable:
                cv2.putText(
                    frame,
                    "Instability Detected!",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                print(
                    "Instability detected: Center of Gravity is misaligned with Base of Support"
                )
                if is_stable:
                    is_stable = False
                    fall_count += 1
            else:
                is_stable = True
                cv2.putText(
                    frame,
                    "Stable",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        out.write(frame)

    cap.release()
    out.release()

    # Use ffmpeg to re-encode the video to the desired format (e.g., MP4)
    reencoded_output_path = os.path.join(
        "temp", "formatted_" + os.path.basename(input_path)
    )
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        output_path,
        "-vcodec",
        "libx264",
        "-acodec",
        "aac",
        "-strict",
        "experimental",
        reencoded_output_path,
    ]
    subprocess.run(ffmpeg_command, check=True)

    # Clean up temporary files
    if os.path.exists(output_path):
        os.remove(output_path)

    return reencoded_output_path


@app.route("/")
def index():
    """
    Root endpoint for the API.

    Returns:
        str: Welcome message.
    """
    return "Welcome to the COG API!"


@app.route("/process_video", methods=["POST"])
def upload_and_process_video():
    """
    Endpoint to upload and process a video file.

    Returns:
        Response: Processed video file or error message.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    # Save the uploaded file temporarily
    input_path = os.path.join("temp", file.filename)
    output_path = os.path.join("temp", "processed_" + file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(input_path)

    try:
        output_file = process_video(input_path, output_path)
        return send_file(output_file, mimetype="video/mp4")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


@app.route("/stability", methods=["POST"])
def stability():
    """
    Endpoint to plot the Center of Gravity (COG) and Base of Support (BOS) on the video
    and detect instability.

    Returns:
        Response: Processed video file with stability analysis or error message.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    # Save the uploaded file temporarily
    input_path = os.path.join("temp", file.filename)
    output_path = os.path.join("temp", "processed_" + file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(input_path)

    try:
        output_file = get_stability(input_path, output_path)
        return send_file(output_file, mimetype="video/mp4")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(e)
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    # Run the Flask app on all available network interfaces
    app.run(host="0.0.0.0", port=5000, debug=True)
