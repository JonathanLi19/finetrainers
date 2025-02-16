import pandas as pd
import os

def remove_zero_frames(input_csv_path, output_csv_path):
    """
    Reads a CSV file, removes rows where num_frames=0, and saves the result to a new CSV file.

    Parameters:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the cleaned CSV file.
    """
    # Check if input file exists
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file '{input_csv_path}' does not exist.")
    
    # Read the CSV file
    print("Reading CSV file...")
    data = pd.read_csv(input_csv_path)
    
    # Check if 'num_frames' column exists
    if 'num_frames' not in data.columns:
        raise ValueError("The input CSV does not contain a 'num_frames' column.")
    
    # Remove rows where num_frames is 0
    print("Filtering rows where num_frames=0...")
    filtered_data = data[data['num_frames'] != 0]
    
    # Save the result to a new CSV file
    print(f"Saving the filtered data to '{output_csv_path}'...")
    filtered_data.to_csv(output_csv_path, index=False)
    print("Done!")

# Example usage
if __name__ == "__main__":
    input_csv = "data/Pexels/whole_data/whole_data_flow_filtered_mask_filtered.csv"  # Replace with your input CSV file path
    output_csv = "data/Pexels/whole_data/whole_data_pure.csv"  # Replace with your desired output file path
    remove_zero_frames(input_csv, output_csv)