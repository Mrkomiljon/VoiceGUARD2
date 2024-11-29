import os
import pandas as pd
import torchaudio
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def prepare_multi_class_dataset(input_csv, output_csv, target_sample_rate=16000):
    """
    Prepare dataset for multi-class classification by resampling and encoding labels.

    Args:
    - input_csv: Path to input dataset CSV.
    - output_csv: Path to save processed dataset.
    - target_sample_rate: Desired sampling rate.

    Returns:
    - None
    """
    # Load the dataset
    data = pd.read_csv(input_csv)
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])

    processed_data = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        file_path = row['file_path']
        label = row['label']

        try:
            # Load and resample audio
            waveform, sample_rate = torchaudio.load(file_path)
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform = resampler(waveform)

            # Save processed audio
            output_path = os.path.splitext(file_path)[0] + "_processed.wav"
            torchaudio.save(output_path, waveform, sample_rate=target_sample_rate)
            
            processed_data.append({'file_path': output_path, 'label': label})
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Save processed dataset
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(output_csv, index=False)
    print(f"Processed dataset saved to {output_csv}")

if __name__ == "__main__":
    prepare_multi_class_dataset("your_data_path/audio_multiclass_dataset.csv", "your_data_path/processed_audio_dataset.csv")
