import os
import cv2
import mediapipe as mp

# Suppress TensorFlow Lite warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

def check_orientation(landmarks):
    """
    Determines orientation (facing forward or sideways) based on shoulder z-depth difference.
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate the z-depth difference between shoulders
    z_diff = abs(left_shoulder.z - right_shoulder.z)

    # Log the z-difference for debugging
    print(f"Z-Depth Difference: {z_diff:.3f}")

    # Threshold to determine orientation
    if z_diff < 0.2:  # Adjust based on testing
        return "Facing Forward"
    else:
        return "Sideways"

def get_stability():
    """
    Analyze video for stability by plotting COG and BOS and detecting instability.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video file.

    Returns:
        str: Path to the re-encoded output video file.
    """
    print("stability process...")
    fall_count = 0
    unstable_frames_count = 0
    is_stable = False
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Cannot open video.")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter.fourcc(*"MP4v")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = frame.shape

        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            if landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility < 0.1 or landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].visibility < 0.1:
                cv2.putText(frame, "Ankle not visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                orientation = check_orientation(results.pose_landmarks.landmark)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                )
                cog_coordinates = calculate_cog(
                    landmarks, image_width, image_height
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

                # Define BOS (e.g., foot region - here using LEFT_FOOT_INDEX)
                base_x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].x
                base_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].y
                
                base_ankle1_x =landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y
                base_ankle2_x =landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y 
                base_foot1_x = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
                base_foot2_x = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y

                # Stability condition
                thresold = 0.05
                stable = (
                    abs(base_foot1_x - base_foot2_x) < thresold or abs(base_ankle1_x - base_ankle2_x) < thresold
                )  # Example threshold

                cv2.putText(frame, f"Orientation: {orientation}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
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
                    unstable_frames_count += 1
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
                    
        cv2.imshow("Real-Time Pose Detection", frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    
    unstable_time = unstable_frames_count / frame_rate
    print("Unstable time: ", unstable_time)

    if fall_count != 0:
        average_instable_time = unstable_time / fall_count
    else:
        average_instable_time = 0
    print("Average instability time: ", average_instable_time)
    


# @app.route("/stability", methods=["POST"])
def stability():
    """
    Endpoint to plot the Center of Gravity (COG) and Base of Support (BOS) on the video
    and detect instability.

    Returns:
        Response: Processed video file with stability analysis or error message.
    """
    try:
        print("Processing stability...")
        if "file" not in request.files:
            print("file not recieved")
            return jsonify({"error": "No file part in the request"}), 400

        print("file recieved", request.files)
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected for uploading"}), 400

        print("creating file locations")
        # Save the uploaded file temporarily
        input_path = os.path.join("temp", file.filename)
        output_path = os.path.join("temp", "processed_" + file.filename)
        os.makedirs("temp", exist_ok=True)
        file.save(input_path)
        try:
            print("Processing stability")
            output_file, unstable_time, average_instable_time = get_stability(input_path, output_path)
            response = send_file(output_file, mimetype="video/mp4")

            # Add additional headers for extra data
            response.headers["X-Unstable-time"] = str(round(unstable_time, 2))
            response.headers["X-Average-unstable-time"] = str(round(average_instable_time, 2))
            response.headers['Access-Control-Expose-Headers']= "X-Average-unstable-time, X-Unstable-time"
            return response
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(e)
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up temporary files
            if os.path.exists(input_path):
                os.remove(input_path)
            # if os.path.exists(output_path):
            #     os.remove(output_path)
    except Exception as e:
        print(e)

get_stability()