import cv2
import os

# Path to the video file
video_path = "backend/temp/formatted_sudo.mp4"

# Directory to save frames
output_folder = f"frames/{os.path.splitext(os.path.basename(video_path))[0]}"
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Frame counter
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # Stop when the video ends

    # Save frame as an image file
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Extracted {frame_count} frames and saved in '{output_folder}'")
