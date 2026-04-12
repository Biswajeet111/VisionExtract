import os
import urllib.request

MODEL_URL = "https://huggingface.co/Biswajeet1/VisionExtract/resolve/main/best_model.pth"

MODEL_PATH = "checkpoints/best_model.pth"

if not os.path.exists(MODEL_PATH):

    os.makedirs("checkpoints", exist_ok=True)

    print("Downloading model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Model downloaded!")