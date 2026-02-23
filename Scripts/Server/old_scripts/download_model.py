# Save this as download_model.py
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print(f"Starting download for {MODEL_ID}...")
# This will save to your default ~/.cache/huggingface/hub/
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype="auto", 
    device_map="cpu" # Download on CPU first to avoid memory issues
)
print("Download complete!")
