import cv2
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Extract frames from a video.')
parser.add_argument('video_path', type=str, help='Path to the video file')
parser.add_argument('output_folder', type=str, help='Folder to save the extracted frames')
parser.add_argument('frequency', type=float, help='Maximum frequency (in Hz) to extract frames')

# Parse the arguments
args = parser.parse_args()

# Define the path to the video, output folder, and extraction frequency
video_path = args.video_path
output_folder = args.output_folder
n_hz = args.frequency

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video's frames per second (fps)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval_ms = 1000 / n_hz  # The interval between frames in milliseconds

print(f"Video FPS is {fps}. Extracting frames at a maximum of {n_hz} Hz.")

frame_count = 0
last_saved_time_ms = 0  # Keeps track of the last saved frame time

# Read and save frames
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Get the current timestamp in milliseconds
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_ns = int(timestamp_ms * 1_000_000) + 1_000_000_000   # Convert milliseconds to nanoseconds
        
        # Only save the frame if enough time has passed since the last saved frame
        if timestamp_ms - last_saved_time_ms >= frame_interval_ms:
            # Save the frame with the current timestamp as the filename
            frame_filename = os.path.join(output_folder, f'{int(timestamp_ns)}.png')
            cv2.imwrite(frame_filename, frame)
            last_saved_time_ms = timestamp_ms
            frame_count += 1
    else:
        break

# Release the video capture object
cap.release()
print(f"Extracted {frame_count} frames to {output_folder}.")

