import os
import pandas as pd

def prepare_multiclass_dataset(gt_folder, ai_folders, ai_labels, output_csv="audio_multiclass_dataset.csv"):
    """
    Prepares a multi-class dataset CSV file for real and AI-generated audio files.

    Args:
    - gt_folder (str): Path to the folder containing real audio (ground truth).
    - ai_folders (list): List of folders containing AI-generated audio.
    - ai_labels (list): List of labels corresponding to AI-generated audio folders.
    - output_csv (str): Name of the output CSV file (default: "audio_multiclass_dataset.csv").

    Returns:
    - None: Saves the dataset as a CSV file.
    """
    data = []

    # Reading real audio files
    print("Reading real audio files...")
    for file in os.listdir(gt_folder):
        if file.endswith(".wav"):  # Include only .wav files
            data.append({'file_path': os.path.join(gt_folder, file), 'label': 'real'})
    
    # Reading AI-generated audio files
    print("Reading AI-generated audio files...")
    for ai_folder, label in zip(ai_folders, ai_labels):
        for file in os.listdir(ai_folder):
            if file.endswith(".wav"):  # Include only .wav files
                data.append({'file_path': os.path.join(ai_folder, file), 'label': label})
    
    # Creating a DataFrame and saving it as a CSV
    print("Saving dataset to CSV format...")
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved: {output_csv}")

# Define folder paths
gt_folder = "./data/gt"  # Folder containing real audio
ai_folders = [
    "./data/diffwave",
    "./data/melgan",
    "./data/parallel_wave_gan",
    "./data/wavegrad",
    "./data/wavenet",
    "./data/wavernn"
]  # Folders containing AI-generated audio
ai_labels = ["diffwave", "melgan", "parallel_wave_gan", "wavegrad", "wavenet", "wavernn"]  # Labels for AI-generated audio

# Prepare the dataset
prepare_multiclass_dataset(gt_folder, ai_folders, ai_labels)
