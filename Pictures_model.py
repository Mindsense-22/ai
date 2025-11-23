import torch
from PIL import Image
from transformers import pipeline

MODEL_NAME = "dima806/facial_emotions_image_detection"

emotion_classifier = pipeline(
    task="image-classification",
    model=MODEL_NAME
)

def map_emotions(results_7: list) -> dict:
    scores = {item["label"].lower(): item["score"] for item in results_7}

    stress_score = scores.get("angry", 0) + scores.get("fear", 0)
    
    fatigue_score = scores.get("sad", 0) + scores.get("disgust", 0)
    
    calm_score = scores.get("neutral", 0) + scores.get("happy", 0)
    

    final_scores = {
        "Anxiety/Stress": stress_score,
        "Fatigue": fatigue_score,
        "Neutral/Calm": calm_score
    }

    dominant_state = max(final_scores, key=final_scores.get)

    return {
        "final_scores": final_scores,
        "dominant_state": dominant_state
    }


def analyze_image_stream(image_input):

    if isinstance(image_input, str):
        try:
            image = Image.open(image_input)
        except FileNotFoundError:
            print(f"⚠️ Picture not found: {image_input}")
            return None
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("Must provide a file path or a PIL.Image object.")

    results_7 = emotion_classifier(image)

    final_analysis = map_emotions(results_7)

    print("\n--- Facial Emotional Analysis ---")
    print(f"Dominant State: {final_analysis['dominant_state']}")
    print("--------------------------------------")
    print("Aggregated Scores:")
    for state, score in final_analysis["final_scores"].items():
        print(f"- {state}: {score:.2f}")

    return final_analysis