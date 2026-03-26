# Dependencies
import bentoml
from PIL import Image
import base64
import torch
import io
import json
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Context-enriched prompts per cube color
CUBE_PROMPTS = {
    "red": [
        "red cube on the floor",
        "small red block on the ground near the robot arm",
        "red colored cube in the scene",
    ],
    "green": [
        "small green cube on the floor, not a vehicle",
        "tiny green block on the ground, not machinery",
        "green colored small cube object on the floor",
    ],
    "blue": [
        "blue cube on the floor",
        "small blue block on the ground",
        "blue colored small cube on the floor near the robot",
    ],
}


@bentoml.service(name="VLM_Service_Isaac", resources={"gpu": 1})
class VLMServiceIsaac:
    def __init__(self):
        self.model_id = MODEL_ID
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
        ).eval()
        print(f"-----MODEL: {self.model_id} LOADED-----")

    def _infer(self, image: Image.Image, instruction: str) -> dict:
        """Run a single VLM inference with given instruction."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": f'Locate the {instruction}. Return only the JSON: {{"target": {{"bbox_xyxy": [ymin, xmin, ymax, xmax]}}}} using normalized coordinates 0-1000.',
                    },
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False, temperature=1.0
            )
            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            out_text = self.processor.batch_decode(
                trimmed_ids, skip_special_tokens=True
            )[0]

        return self.extract_coords(out_text)
    
    def _is_round_number_bbox(self, coords):
        return all(c % 50 == 0 for c in coords)

    @bentoml.api
    async def ground(self, instruction: str, image_b64: str) -> dict:
        """Single prompt detection — kept for backwards compatibility."""
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self._infer(image, instruction)

    @bentoml.api
    async def ground_multi(self, color: str, image_b64: str) -> dict:
        """
        Multi-prompt detection for a given cube color.
        Runs all prompts for that color, returns median bbox of valid detections.
        
        Args:
            color: "red", "green", or "blue"
            image_b64: base64 encoded PNG image
            
        Returns:
            {
                "target": {
                    "found": True/False,
                    "bbox_xyxy": [ymin, xmin, ymax, xmax],  # median of valid detections
                    "num_valid": int,   # how many prompts agreed
                    "all_bboxes": [...] # all valid detections for debugging
                }
            }
        """
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        prompts = CUBE_PROMPTS.get(color.lower(), [f"{color} cube on the floor"])

        valid_bboxes = []
        for prompt in prompts:
            result = self._infer(image, prompt)
            if result and result.get("target") and result["target"].get("found"):
                valid_bboxes.append(result["target"]["bbox_xyxy"])

        if len(valid_bboxes) == 0:
            return {"target": {"found": False, "num_valid": 0}}

        # In ground_multi, filter outliers before median:
        if len(valid_bboxes) >= 2:
            sizes = [abs(b[2]-b[0]) * abs(b[3]-b[1]) for b in valid_bboxes]
            median_size = np.median(sizes)
            valid_bboxes = [b for b, s in zip(valid_bboxes, sizes) 
                            if abs(s - median_size) < median_size * 0.5]

        # Compute median bbox across all valid detections
        bboxes_arr = np.array(valid_bboxes)
        median_bbox = np.median(bboxes_arr, axis=0).astype(int).tolist()

        return {
            "target": {
                "found": True,
                "bbox_xyxy": median_bbox,
                "num_valid": len(valid_bboxes),
                "all_bboxes": valid_bboxes,
            }
        }

    @bentoml.api
    async def find_jetbot(self, image_b64: str) -> dict:
        """Detect the Jetbot wheeled robot vehicle."""
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": 'Locate the small green robot vehicle on wheels. Return only the JSON: {"target": {"bbox_xyxy": [ymin, xmin, ymax, xmax]}} using normalized coordinates 0-1000.',
                    },
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False, temperature=1.0
            )
            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            out_text = self.processor.batch_decode(
                trimmed_ids, skip_special_tokens=True
            )[0]

        return self.extract_coords(out_text)

    def extract_coords(self, text):
        """Extract bbox coordinates from VLM output with hallucination filters."""
        bracket_match = re.search(
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text
        )

        if bracket_match:
            coords = [int(x) for x in bracket_match.groups()]
            # Hallucination filters
            if coords[0] < 10 and coords[1] < 10:
                return {"target": {"found": False}}
            if self._is_round_number_bbox(coords):
                return {"target":{"found": False}}
            if (coords[2] - coords[0]) < 20 or (coords[3] - coords[1]) < 20:
                return {"target": {"found": False}}
            return {"target": {"bbox_xyxy": coords, "found": True}}
        return {"target": {"found": False}}
    