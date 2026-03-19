import torch
from PIL import Image
from transformers import pipeline
import io

device = 0 if torch.cuda.is_available() else -1
MODEL_NAME = "dima806/facial_emotions_image_detection"

print(f"🔄 Loading Face Model on {'GPU' if device == 0 else 'CPU'}...")
emotion_classifier = pipeline(
    task="image-classification", model=MODEL_NAME, device=device, top_k=None
)


def map_emotions(results_7: list) -> dict:
    scores = {item["label"].lower(): item["score"] for item in results_7}

    anger_score = scores.get("angry", 0) + scores.get("fear", 0)

    sadness_score = scores.get("sad", 0) + scores.get("disgust", 0)

    happy_score = scores.get("happy", 0) + scores.get("surprise", 0)

    neutral_score = scores.get("neutral", 0)

    final_scores = {
        "Angry": anger_score,
        "Sad": sadness_score,
        "Happy": happy_score,
        "Neutral": neutral_score,
    }

    dominant_state = max(final_scores, key=final_scores.get)
    # return dominant_state
    return final_scores


def analyze_face_stream(image_input):
    try:
        image = None

        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str):
            image = Image.open(image_input)

        if image.mode != "RGB":
            image = image.convert("RGB")

        results = emotion_classifier(image)
        result_state = map_emotions(results)
        return result_state

    except Exception as e:
        print(f"❌ Face Error: {e}")

        return "Neutral"
