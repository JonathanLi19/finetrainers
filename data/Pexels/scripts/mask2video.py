import os
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def frames_to_video(frames_path: str, output_video_path: str, fps: int = 30, frame_ext: str = '.png'):
    """
    Converts frames from a folder into a video.

    Args:
        frames_path (str): Path to the folder containing the frames.
        output_video_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the output video.
        frame_ext (str): Extension of the frame files (default is '.png').
    """
    # Get the list of all frame files in the folder
    frame_files = sorted([os.path.join(frames_path, f) for f in os.listdir(frames_path) if f.endswith(frame_ext)])

    # Ensure there are frames in the folder
    if not frame_files:
        raise RuntimeError(f"No frames found in the directory: {frames_path}")

    # Read the first frame to get the frame size
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For saving in .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Iterate over all frames and write them to the video
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)  # Write the frame to the video

    # Release the VideoWriter object
    out.release()
    print(f"Video saved at {output_video_path}")


def process_single_trajectory(frames_path: str, output_video_path: str, fps: int, frame_ext: str):
    """
    Processes a single trajectory map and converts frames to a video.
    """
    if os.path.exists(output_video_path):
        print(f"Video already exists for {frames_path}, skipping...")
        return

    if os.path.exists(frames_path) and os.path.isdir(frames_path):
        frames_to_video(frames_path, output_video_path, fps, frame_ext)
    else:
        print(f"Directory not found: {frames_path}")


def process_trajectory_maps(csv_file: str, fps: int = 30, frame_ext: str = '.png', num_workers: int = None):
    """
    Processes trajectory maps by converting the frames in each trajectory_maps_path into a video.

    Args:
        csv_file (str): Path to the CSV file containing the trajectory_maps_path.
        fps (int): Frames per second for the output videos.
        frame_ext (str): Extension of the frame files (default is '.png').
        num_workers (int): Number of worker threads for parallel processing (default is None, which uses the system default).
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check if the required column exists
    if 'trajectory_maps_path' not in df.columns:
        raise ValueError("CSV file must contain 'trajectory_maps_path' column.")

    # Use a ThreadPoolExecutor with the specified number of workers
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {}

        # Iterate over each trajectory_maps_path in the DataFrame
        for index, row in df.iterrows():
            frames_path = row['trajectory_maps_path']

            # Determine the output video path (parent directory of frames_path)
            output_video_path = os.path.join(os.path.dirname(frames_path), f"masks.mp4")

            # Submit the processing task to the executor
            future = executor.submit(process_single_trajectory, frames_path, output_video_path, fps, frame_ext)
            future_to_path[future] = frames_path

        # Optionally, handle completed futures
        for future in as_completed(future_to_path):
            frames_path = future_to_path[future]
            try:
                future.result()  # Get the result of the future
            except Exception as e:
                print(f"Error processing {frames_path}: {e}")


# Example usage:
csv_file_path = '/home/qid/quanhao/workspace/Open-Sora/data/Pexels/data_mini.csv'  # Path to your CSV file
process_trajectory_maps(csv_file_path, fps=30, num_workers=96)  # Set num_workers to 4