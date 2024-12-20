# AI vs. Human Voice Classification Using Wav2Vec2
<p align="center">
  <img src="https://github.com/Mrkomiljon/VoiceGUARD2/blob/main/logs/image-71.webp" alt="Image 1" width="400" height="200">
  <img src="https://github.com/Mrkomiljon/VoiceGUARD2/blob/main/logs/cc.jpg" alt="Image 2" width="400" height="200">
  </p>
  
## 🌟 Overview 🌟

This project provides an end-to-end solution for classifying audio into **human** or **AI-generated** categories using the **Wav2Vec2** model. It supports multi-class classification and distinguishes between real voices and synthetic audio generated by models like **DiffWave**, **WaveNet**, and more. The pipeline includes:

- **Dataset Preparation**: Combine human and AI-generated audio into a structured, multi-class dataset.
- **Preprocessing**: Resample audio, normalize lengths, and Convert to HuggingFace Dataset format.
- **Fine-Tuning**: Adapt Wav2Vec2 for custom classification tasks.
- **Inference**: Classify unseen audio with confidence scores.
- **API Deployment**: Real-time predictions via FastAPI.

---

## 🚀 Key Features

- **Multi-Class Audio Classification**: Supports detection of human voices and six AI-generated classes (DiffWave, WaveNet, etc.).
- **Optimized for Performance**: Includes techniques like attention masks, quantization, and ONNX conversion for efficiency.
- **Deployment-Ready**: Real-time audio classification using FastAPI.
## Available on Hugging Face 🤗

This model is also hosted on Hugging Face for easy access and inference:

[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-blue?logo=huggingface)](https://huggingface.co/Mrkomiljon/voiceGUARD)

## DEMO
![ezgif com-animated-gif-maker (2)](https://github.com/user-attachments/assets/d807785b-1a14-4d6e-88be-9f3c8a4b9b93)

If you find this project helpful or inspiring, please consider giving it a star 🌟 on GitHub! Your support motivates us to keep improving and sharing our work with the community. 😊

## Dataset

### **Source**
The dataset is derived from the **LibriSeVoc** corpus, which includes real human audio samples and synthetic speech generated by:
- **DiffWave**
- **WaveNet**
- **MelGAN**
- **Parallel WaveGAN**
- **WaveGrad**
- **WaveRNN**

### **Structure**
- Real human voices are stored in the `gt` folder.
- AI-generated voices are organized into separate folders (`diffwave`, `wavenet`, etc.).
- Dataset includes **13,201 samples** per class, ensuring balance.

## 📂 File Structure
* ├── prepare_data.py # Combine raw audio into a CSV dataset 
* ├── process_audio.py # Resample, pad, and preprocess audio
* ├── prepare_hf_dataset.py # Convert to HuggingFace Dataset format
* ├── fine-tune.py # Fine-tune Wav2Vec2 on the dataset 
* ├── inference.py # Perform inference on unseen audio 
* ├── app.py # FastAPI app for real-time classification 
* ├── README.md # Documentation

📚 Acknowledgements
* [LibriSeVoc](https://drive.google.com/file/d/1NXF9w0YxzVjIAwGm_9Ku7wfLHVbsT7aG/view) Dataset for providing audio samples.
* [PyTorch](https://pytorch.org/) and Torchaudio for efficient deep learning and audio processing capabilities.
* [Download](https://github.com/Mrkomiljon/VoiceGUARD2/releases/download/version.0.0.1/voiceGUARD2.7z) pre-trained model

  ## 🔧 Setup Instructions

### **Step 1: Clone Repository**
```bash
https://github.com/Mrkomiljon/VoiceGUARD2.git
cd VoiceGUARD2
```
### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```
### **Step 3: Dataset Preparation**
## Run data_preparation.py to prepare a multi-class dataset:

```bash
python prepare_data.py
```
## This script combines real and AI-generated audio into a structured CSV file.

### **Step 4: Preprocessing**
Run preprocess_dataset.py to resample, pad, and preprocess the dataset:

```bash
python process_audio.py
```
### Convert to HuggingFace Dataset format
```bash
python prepare_hf_dataset.py
```
### **Step 5: Fine-Tuning**
Fine-tune the Wav2Vec2 model using fine-tune.py:

```bash
python fine-tune.py
```
### **Step 6: Inference**
Use inference.py to classify unseen audio:

```bash
python inference.py --audio_file path_to_audio.wav
```
### **Step 7: Deploy API**
## Run app.py to start the FastAPI server:

```bash
uvicorn app:app --reload
```
## Access the API at http://127.0.0.1:8000/docs
![ezgif com-animated-gif-maker (3)](https://github.com/user-attachments/assets/23273135-6319-4d60-8b6e-7c1df84ccc87)

# Technical Details
## Model
### The project uses Wav2Vec2ForSequenceClassification fine-tuned on a multi-class dataset of real and AI-generated voices.
## Preprocessing
* Audio is resampled to 16kHz.
* Attention masks and padding are applied for consistent input length.
## Fine-Tuning
The model is fine-tuned for 10 epochs with a learning rate of 2e-5.
Techniques like gradient checkpointing and mixed precision are enabled to improve efficiency.
## Inference
Inference is optimized with ONNX quantization for deployment on edge devices(soon).
## Results
* Validation Accuracy: Achieved 99.8% accuracy on the test set.
* Robustness: The model generalizes well across synthetic and human voices.
## 🔮 Future Improvements
- Add support for more AI-generated voice datasets.
- Test inference on mobile devices using ONNX.
- Implement real-time streaming support for API.
## 📜 License
This project is licensed under the MIT License.




