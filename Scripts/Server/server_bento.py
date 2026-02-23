# Dependencies
import bentoml
from PIL import Image
import base64
import torch
import io
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

@bentoml.service(name="VLM_Service_Isaac", resources={"gpu": 1})
class VLMServiceIsaac:
    def __init__(self):
        self.model_id = MODEL_ID
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    device_map="auto"
                ).eval()
        print(f"-----MODEL: {self.model_id} LOADED-----")

    @bentoml.api
    async def ground(self, instruction: str, image_b64: str) -> dict:
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Locate the {instruction}. Return only the JSON: {{\"target\": {{\"bbox_xyxy\": [ymin, xmin, ymax, xmax]}}}} using normalized coordinates 0-1000."}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            out_text = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]
        
        return self.extract_coords(out_text)

    def extract_coords(self, text):
        # Improved extraction for Qwen's specific bracket style
        bracket_match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text)
        if bracket_match:
            coords = [int(x) for x in bracket_match.groups()]
            return {"target": {"bbox_xyxy": coords, "found": True}}
        return {"target": {"found": False}}
