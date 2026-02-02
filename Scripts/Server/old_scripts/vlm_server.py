from fastapi import FastAPI
from pydantic import BaseModel
import base64, io, re, json
from PIL import Image
import torch

# BYPASSING THE BROKEN __init__.py COMPLETELY
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.idefics3.modeling_idefics3 import Idefics3ForConditionalGeneration

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

app = FastAPI()

class GroundRequest(BaseModel):
    instruction: str
    image_b64: str

# ---- Load model once at startup ----
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model (Idefics3 architecture)...")
model = Idefics3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    _attn_implementation="eager", 
    device_map="auto"
)
model.eval()
print("Model ready!")

def _extract_json(text: str):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except:
        return None

@app.post("/ground")
def ground(req: GroundRequest):
    img_bytes = base64.b64decode(req.image_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    prompt = (
        f"<|user|>\n<|image_1|>\n"
        f"Find the {req.instruction}. "
        "Provide the bounding box in [ymin, xmin, ymax, xmax] format using 0-1000 coordinates. "
        "Output JSON: {\"target\": {\"bbox_xyxy\": [ymin, xmin, ymax, xmax]}}\n<|assistant|>\n"
	)

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda", torch.float16)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    out_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    js = _extract_json(out_text)
    if js and js.get("target") and js["target"].get("bbox_xyxy"):
        bbox = js["target"]["bbox_xyxy"]
        js["target"]["bbox_xyxy"] = [
            int(bbox[0] * 640 / 1000), int(bbox[1] * 480 / 1000),
            int(bbox[2] * 640 / 1000), int(bbox[3] * 480 / 1000)
        ]

    return js or {"action": "pick", "target": None, "raw": out_text}
