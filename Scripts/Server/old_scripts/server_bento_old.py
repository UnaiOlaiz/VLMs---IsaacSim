# Dependencies
import bentoml
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import json
import base64
import torch
import io
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.idefics3.modeling_idefics3 import Idefics3ForConditionalGeneration
import re

# We will define the model id as a variable if we were to change the model
# MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct" # It worked with this one, but did not get pretty good coordinates (allucinated a bit)

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Request class
class VLMRequest(BaseModel):
    '''
    Request will consist of Image-text pairs
    Image: base64 format
    Text: string
    '''
    instruction: str = Field(default="find the red cube")
    image_64: str = Field(description="Base 64 image")

# Model service definition
@bentoml.service(name="VLM_Service_Isaac", resources={"gpu": 1})
class VLMServiceIsaac:
    '''
    Service which will host the different VLMs that we will
    use to analyze the Isaac environment. 
    It will receive an Image (env) - text (prompt) pair and
    it will return the response in a JSON format.
    '''
    def __init__(self):
        self.model_id = MODEL_ID
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            _attn_implementation="eager", 
            device_map="auto"
            )
        self.model.eval()
        print("\n" + "-" * 50 + f" MODEL: {MODEL_ID.upper()} LOADED TO THE BENTOML SERVICE" + "-" * 50 + "\n")

    # API interaction function
    @bentoml.api
    def ground(self, instruction: str, image_b64: str) -> dict:
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        width, height = image.size

        # Template structure: few-shot structured learning
        messages = [
            {
            "role": "user",
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": "Detect the red cube. Output only the bounding box as [ymin, xmin, ymax, xmax] using normalized coordinates 0-1000."}
            ]
        }
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to("cuda", torch.float16)

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        out_text = self.processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]        
        print(f"--- RAW MODEL OUTPUT: {out_text} ---") 
        
        js = self.extract_json(out_text)

        # FIX FOR VALIDATION ERROR: Always return a dict, even if empty
        if js is None or "target" not in js:
            print("Model failed to find coordinates or JSON format.")
            return {"target": {"bbox_xyxy": [0, 0, 0, 0], "found": False}}

        # Logic for pixel conversion
        raw_bbox = js["target"].get("bbox_xyxy", [0, 0, 0, 0])
        js["target"]["raw_norm_bbox"] = raw_bbox 
        js["target"]["bbox_pixels"] = [
            int(raw_bbox[0] * height / 1000), 
            int(raw_bbox[1] * width / 1000),
            int(raw_bbox[2] * height / 1000), 
            int(raw_bbox[3] * width / 1000)
        ]
        js["target"]["found"] = True
        return js
    
    def extract_json(self, text):

        if not text:
            return None

        print(f"\nAggressive extraction from: {text}\n")

        # Step 1: Strict JSON search
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass

        # Step 2: Bracket search [y1, x1, y2, x2]
        # This handles [20.0, 20.0, 10.0, 10.0] or [20, 20, 10, 10]
        bracket_match = re.search(r"\[\s*([\d\.]+)\s*[,;]\s*([\d\.]+)\s*[,;]\s*([\d\.]+)\s*[,;]\s*([\d\.]+)\s*\]", text)
        if bracket_match:
            try:
                coords = [int(float(x)) for x in bracket_match.groups()]
                return {"target": {"bbox_xyxy": coords}}
            except ValueError:
                pass

    # Step 3: "Desperation" search - find ANY 4 numbers in the text
    # This handles "The box is at 20 20 10 10" or "Points: 20.5, 20.5, 10.1, 10.2"
        all_numbers = re.findall(r"[\d\.]+", text)
        if len(all_numbers) >= 4:
            try:
            # We take the first 4 numbers found
                coords = [int(float(x)) for x in all_numbers[:4]]
                # Sanity check: coordinates shouldn't all be 0
                if sum(coords) > 0:
                    return {"target": {"bbox_xyxy": coords}}
            except ValueError:
                pass

        return None