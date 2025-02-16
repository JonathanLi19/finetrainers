import pandas as pd
import os

def sample_random_rows(input_csv: str, output_csv: str, sample_size: int = 100):
    """
    Randomly sample rows from a CSV file and save the result to a new file.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file where the sampled data will be saved.
        sample_size (int): The number of rows to sample. Default is 100.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Check if the dataset has fewer rows than the sample size
    if len(df) < sample_size:
        raise ValueError(
            f"The dataset has only {len(df)} rows, which is fewer than the requested sample size {sample_size}.")

    # Randomly sample the rows
    sampled_df = df.sample(n=sample_size, random_state=42)

    # Save the sampled rows to a new CSV file
    sampled_df.to_csv(output_csv, index=False)
    print(f"Sampled {sample_size} rows and saved to {output_csv}")


# # Example usage:
# input_csv = '/home/qid/quanhao/workspace/Open-Sora/data/Pexels/data.csv'
# output_csv = '/home/qid/quanhao/workspace/Open-Sora/data/Pexels/data_mini.csv'
# sample_random_rows(input_csv, output_csv, sample_size=100)


def add_video_path_column(csv_file: str):
    """
    Reads a CSV file, adds a trajectory_video_path column, and saves the result back to the same CSV file.

    Args:
        csv_file (str): Path to the input CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check if the required column exists
    if 'trajectory_maps_path' not in df.columns:
        raise ValueError("CSV file must contain 'trajectory_maps_path' column.")

    # Create the trajectory_video_path column
    df['trajectory_video_path'] = df['trajectory_maps_path'].apply(
        lambda x: os.path.join(os.path.dirname(x), "masks.mp4")
    )

    # Save the updated DataFrame back to the same CSV file
    df.to_csv(csv_file, index=False)
    print(f"Updated CSV saved to {csv_file}")


# Example usage:
csv_path = '/home/qid/quanhao/workspace/Open-Sora/data/Pexels/data_mini.csv'  # Path to your input CSV file
add_video_path_column(csv_path)
