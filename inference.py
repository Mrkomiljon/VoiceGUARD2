import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torchaudio

# Load the fine-tuned model and processor
model_path = "your_model_path/wav2vec2_finetuned_model"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
processor = Wav2Vec2Processor.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# Device setup (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_audio(audio_path, target_sample_rate=16000, temperature=1.0):
    """
    Classify an audio file using the fine-tuned Wav2Vec2 model.

    Args:
    - audio_path (str): Path to the audio file.
    - target_sample_rate (int): Target sample rate for the model.
    - temperature (float): Temperature for softmax scaling (default=1.0 for no scaling).

    Returns:
    - label (int): Predicted numerical label for the audio file (0-6 for 7 classes).
    - confidence (float): Confidence score for the prediction.
    - logits (list): Raw model logits for all classes.
    """
    try:
        # Load and resample audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Ensure audio length is consistent with training (10 seconds)
        max_length = target_sample_rate * 10  # 10 seconds in samples
        if waveform.size(1) > max_length:
            waveform = waveform[:, :max_length]  # Truncate
        elif waveform.size(1) < max_length:
            waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.size(1)))  # Pad

        # Use only the first channel if the waveform has multiple channels
        if waveform.ndim > 1:
            waveform = waveform[0]

        # Process audio
        inputs = processor(waveform.numpy(), sampling_rate=target_sample_rate, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Perform inference
        with torch.no_grad():
            logits = model(**inputs).logits
            print(f"Logits shape: {logits.shape}")  # Debugging logits shape
            probabilities = torch.nn.functional.softmax(logits / temperature, dim=-1)
            predicted_label = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_label].item()

        return predicted_label, confidence, logits.tolist()
    except Exception as e:
        print(f"Error processing file {audio_path} - {e}")
        return None, None, None

# Example usage
if __name__ == "__main__":
    audio_file_path = "your_data_path/audios/clova.mp3"  # Replace with your audio file
    label, confidence, logits = classify_audio(audio_file_path, temperature=2)  # Adjust temperature if needed
    if label is not None:
        print(f"Predicted Label: {label}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Logits: {logits}")
    else:
        print("Failed to classify the audio file.")
