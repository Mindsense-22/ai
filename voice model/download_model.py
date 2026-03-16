from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch

# إعداد كارت الشاشة
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

model_name = "superb/wav2vec2-base-superb-er"
print("⏳ Loading Model from SafeTensors...")

try:
    # 1. تحميل مستخرج الخصائص
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    
    # 2. تحميل الموديل (مع إجبار استخدام النسخة الآمنة لتخطي الخطأ)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name, 
        use_safetensors=True  # <--- دي الإضافة السحرية
    )
    
    model.to(device)
    
    print("="*50)
    print(f"✅ SUCCESS! Model loaded cleanly on: {device}")
    print("😎 Emotions recognized:")
    print(list(model.config.id2label.values()))
    print("="*50)

except Exception as e:
    print(f"❌ Error: {e}")