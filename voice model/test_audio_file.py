import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import os

# 1. تجهيز كارت الشاشة
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# 2. تحميل الموديل
model_name = "superb/wav2vec2-base-superb-er"
print("⏳ Loading Model...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
model.to(device)
print("✅ Model Ready!")

# ==========================================
# 3. اسم الملف اللي هنجربه (غير الاسم هنا لو مختلف)
filename = "test_audio.wav" 
# ==========================================

def predict_emotion(filepath):
    # تحميل الصوت
    waveform, sample_rate = torchaudio.load(filepath)
    
    # أهم خطوة: توحيد التردد لـ 16000 (الموديل مبيفهمش غير كده)
    if sample_rate != 16000:
        print(f"⚠️ Resampling from {sample_rate}Hz to 16000Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # لو الصوت ستيريو (قناتين)، خليه مونو (قناة واحدة)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # تجهيز الداتا للموديل
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    
    # التوقع
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # استخراج النتيجة
    pred_id = torch.argmax(logits, dim=-1).item()
    emotion = model.config.id2label[pred_id]
    
    return emotion

# --- التشغيل ---
if os.path.exists(filename):
    print(f"📂 Analyzing file: {filename} ...")
    try:
        result = predict_emotion(filename)
        
        print("\n" + "="*40)
        print(f"🎉 THE AI SAYS:  👉  {result.upper()}  👈")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print(f"\n❌ Error: الملف '{filename}' مش موجود!")
    print("تأكد إنك حطيت ملف الصوت جوه الفولدر وغيرت اسمه في الكود.")