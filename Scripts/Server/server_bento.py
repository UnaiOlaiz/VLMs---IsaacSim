# BentoML VLM Service — two-stage platform detection
# Stage 1 (discover): find all platforms in the scene
# Stage 2 (classify): confirm color of a single cropped platform

import bentoml
from PIL import Image
import base64
import torch
import io
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


@bentoml.service(name="VLM_Service_Isaac", resources={"gpu": 1})
class VLMServiceIsaac:
    def __init__(self):
        self.model_id = MODEL_ID
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        ).eval()
        print(f"-----MODEL: {self.model_id} LOADED-----")

    # ── helpers ──────────────────────────────────────────────────────────

    def _decode_image(self, image_b64: str):
        img_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _run_model(self, messages, max_new_tokens=256):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0
            )
            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        return self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

    def _is_valid_bbox(self, coords):
        """Reject hallucinated tiny or corner bboxes."""
        ymin, xmin, ymax, xmax = coords
        if (ymax - ymin) < 20 or (xmax - xmin) < 20:
            return False
        if ymin < 10 and xmin < 10:
            return False
        return True

    def _extract_all_bboxes(self, text):
        """Extract all valid bboxes from VLM output."""
        all_bboxes = re.findall(
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text
        )
        valid = []
        for b in all_bboxes:
            coords = [int(x) for x in b]
            if self._is_valid_bbox(coords):
                valid.append(coords)
        return {"platforms": valid}

    # ── Stage 1: discover all platforms ──────────────────────────────────

    @bentoml.api
    async def discover(self, image_b64: str) -> dict:
        """Find all flat colored platforms in the scene, return all bboxes."""
        image = self._decode_image(image_b64)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "Find all flat colored platforms on the floor in this image. "
                    "Return a JSON list: "
                    "{\"platforms\": [[ymin,xmin,ymax,xmax], ...]} "
                    "using normalized coordinates 0-1000. Return only the JSON."
                )},
            ],
        }]
        out_text = self._run_model(messages)
        print(f"Discover output: {out_text}")
        return self._extract_all_bboxes(out_text)

    # ── Stage 2: classify a single platform crop ──────────────────────────

    @bentoml.api
    async def classify(self, instruction: str, image_b64: str) -> dict:
        """Given a crop of one platform, confirm if it matches the requested color."""
        image = self._decode_image(image_b64)
        color = instruction.strip().lower().split()[0]  # "green", "red", or "blue"
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Is the platform in this image {color}? "
                    f"Answer only YES or NO."
                )},
            ],
        }]
        out_text = self._run_model(messages, max_new_tokens=8)
        print(f"Classify output: {out_text}")
        answered_yes = "yes" in out_text.lower()
        return {"match": answered_yes}

    # ── Legacy endpoint (kept for compatibility) ──────────────────────────

    @bentoml.api
    async def ground(self, instruction: str, image_b64: str) -> dict:
        """Original single-shot grounding endpoint."""
        image = self._decode_image(image_b64)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Top-down camera view. Three colored floor platforms: "
                    f"GREEN at top-center, BLUE at left, RED at right. "
                    f"Locate ONLY the {instruction}. "
                    f"Return only JSON: {{\"target\": {{\"bbox_xyxy\": [ymin,xmin,ymax,xmax]}}}} "
                    f"in 0-1000 coordinates."
                )},
            ],
        }]
        out_text = self._run_model(messages)
        print(f"Ground output: {out_text}")
        match = re.search(
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", out_text
        )
        if match:
            coords = [int(x) for x in match.groups()]
            if self._is_valid_bbox(coords):
                return {"target": {"bbox_xyxy": coords, "found": True}}
        return {"target": {"found": False}}