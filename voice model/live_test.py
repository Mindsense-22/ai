import torch
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torchaudio
import numpy as np
import os

# 1. تجهيز الموديل
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

model_name = "superb/wav2vec2-base-superb-er"
print("⏳ Loading AI Model...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
model.to(device)
print("✅ Model Ready!")

# 2. دالة التسجيل
def record_audio(duration=4, fs=16000):
    print("\n" + "="*40)
    print(f"🎤 Speak NOW! (Recording for {duration}s)...")
    print("="*40)
    
    # بدء التسجيل
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # انتظار انتهاء التسجيل
    print("⏹️ Finished.")
    
    # حفظ الملف
    filename = "test_voice.wav"
    write(filename, fs, (myrecording * 32767).astype(np.int16))
    return filename

# 3. دالة التحليل
def predict_emotion(filename):
    waveform, sample_rate = torchaudio.load(filename)
    
    # توحيد التردد لـ 16000
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # التوقع
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    pred_id = torch.argmax(logits, dim=-1).item()
    return model.config.id2label[pred_id]

# --- التنفيذ ---
if __name__ == "__main__":
    try:
        file = record_audio()
        print("🧠 Analyzing...")
        emotion = predict_emotion(file)
        
        print("\n" + "*"*40)
        print(f"🎉 Result: {emotion.upper()}") 
        print("*"*40 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")