import pandas as pd
from datasets import Dataset, Audio
from transformers import Wav2Vec2Processor
import numpy as np

def load_and_prepare_dataset(processed_csv, target_sample_rate=16000, max_length=10.0):
    """
    Load and prepare the processed dataset for Wav2Vec2 fine-tuning.

    Args:
    - processed_csv: Path to the processed CSV file with audio paths and labels.
    - target_sample_rate: Target sample rate for audio files.
    - max_length: Maximum audio length in seconds.

    Returns:
    - A prepared dataset split into training and testing sets.
    """
    # Load the dataset
    data = pd.read_csv(processed_csv)

    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_pandas(data)

    # Map audio paths to the dataset
    dataset = dataset.cast_column("file_path", Audio(sampling_rate=target_sample_rate))

    # Define processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    # Preprocess the audio and extract input values
    def preprocess_audio(batch):
        max_input_length = int(target_sample_rate * max_length)  # Convert max_length to number of samples
        inputs = []

        for audio_file in batch["file_path"]:
            audio = audio_file["array"]
            
            # Truncate or pad audio to fixed length
            if len(audio) > max_input_length:
                audio = audio[:max_input_length]
            elif len(audio) < max_input_length:
                audio = np.pad(audio, (0, max_input_length - len(audio)))

            # Process audio
            inputs.append(processor(audio, sampling_rate=target_sample_rate, return_tensors="pt").input_values[0])
        
        batch["input_values"] = inputs
        return batch

    # Apply preprocessing to the dataset
    dataset = dataset.map(
        preprocess_audio, 
        remove_columns=["file_path"], 
        batched=True
    )

    return dataset

# Prepare the dataset
if __name__ == "__main__":
    prepared_dataset = load_and_prepare_dataset(
        "your_data_path/processed_audio_dataset.csv",
        target_sample_rate=16000,
        max_length=10.0  # Adjust this value based on your dataset
    )

    # Split into training and testing sets
    dataset_split = prepared_dataset.train_test_split(test_size=0.2)

    # Save the dataset
    dataset_split.save_to_disk("your_data_path/prepared_dataset")
    print("Dataset has been prepared and saved.")
