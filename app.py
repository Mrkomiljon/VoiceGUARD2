from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch
import torchaudio
from io import BytesIO
from pathlib import Path

# Load model and processor
model_path = "C:\\Users\\GOOD\\Desktop\\TEST-2024\\transformer\\wav2vec2_finetuned_model"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, local_files_only=True)
processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
model.eval()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# FastAPI app
app = FastAPI()

# Set up templates
templates = Jinja2Templates(directory="C:\\Users\\GOOD\\Desktop\\TEST-2024\\transformer\\templates")
# define label mapping
id2label = {
    0: "diffwave",
    1: "melgan",
    2: "parallel_wave_gan",
    3: "Real",
    4: "wavegrad",
    5: "wavnet",
    6: "wavernn"
}

class PredictionResponse(BaseModel):
    label: int
    class_name: str
    confidence: float


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    target_sample_rate = 16000  # Model's expected sample rate
    max_length = target_sample_rate * 10  # 10 seconds in samples

    try:
        # Load the audio file
        audio_bytes = await file.read()
        waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))

        # Resample if the sample rate doesn't match the model's expected rate
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Truncate or pad the waveform to ensure consistent input length
        if waveform.size(1) > max_length:
            waveform = waveform[:, :max_length]  # Truncate
        elif waveform.size(1) < max_length:
            waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.size(1)))  # Pad
        if waveform.ndim > 1:
            waveform = waveform[0]

        # Process the audio file
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=target_sample_rate,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs["input_values"].to(device)

        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_label].item()

        # Map label to class name
        class_name = id2label.get(predicted_label, "Unknown Class")

        return {
            "label": int(predicted_label),
            "class_name": class_name,
            "confidence": float(confidence)
        }
    except Exception as e:
        return {"error": f"Error processing the audio file: {str(e)}"}