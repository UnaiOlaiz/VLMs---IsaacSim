import bentoml
from PIL import Image
import base64
import torch
import io
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

CUBE_PROMPTS = {
    "red": ["red cube in the center", "small red block near robot"],
    "green": ["small green cube on the right side", "green block on the right"],
    "blue": ["small blue cube on the left side", "blue block on the left"],
}

PALLET_PROMPTS = {
    "red": [
        "red plastic pallet on the floor",
        "top view of a red cargo pallet near the robot",
        "the center of the red rectangular grid structure",
    ],
    "blue": [
        "blue plastic pallet located to the left",
        "blue industrial pallet on the grey surface",
        "top-down view of a blue rectangular pallet",
    ],
    "black": [
        "black plastic pallet on the right side",
        "dark grey cargo pallet on the floor",
        "rectangular black grid structure",
    ],
}


@bentoml.service(name="VLM_Service_Isaac", resources={"gpu": 1})
class VLMServiceIsaac:
    def __init__(self):
        self.model_id = MODEL_ID
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="cuda",
        ).eval()
        print(f"########## MODEL: {MODEL_ID.upper()} LOADED ##########")

    def _is_hallucination(self, coords):
        ymin, xmin, ymax, xmax = coords
        width, height = xmax - xmin, ymax - ymin
        if width > 600 or height > 600 or width < 8 or height < 8:
            return True
        return False

    def _infer(self, image: Image.Image, instruction: str) -> dict:
        prompt = (
        f"Locate the {instruction}. I have marked its exact color center with a yellow square. "
        f"Focus ONLY on the center of that yellow square to provide the coordinates. "
        f"Ignore all black shadows or robot parts. "
        f"Return only JSON: {{\"target\": {{\"bbox_xyxy\": [ymin, xmin, ymax, xmax]}}}} "
        f"using normalized coordinates 0-1000."
        )
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": f'Locate the {instruction}. Return only JSON: {{"target": {{"bbox_xyxy": [ymin, xmin, ymax, xmax]}}}} using normalized coordinates 0-1000.',
                    },
                ],
            }
        ]
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False
            )
            trimmed_ids = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            out_text = self.processor.batch_decode(
                trimmed_ids, skip_special_tokens=True
            )[0]

        bracket_match = re.search(
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", out_text
        )
        if bracket_match:
            coords = [int(x) for x in bracket_match.groups()]
            if self._is_hallucination(coords):
                return {"target": {"found": False}}
            return {"target": {"bbox_xyxy": coords, "found": True}}
        return {"target": {"found": False}}

    @bentoml.api
    async def ground_multi(
        self,
        color: str,
        image_b64: str,
        target_type: str = "cube",
        custom_prompt: str = None,
    ) -> dict:
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if custom_prompt:
            print(f"\nVLM: using custom prompt: {custom_prompt}")
            return self._infer(image, custom_prompt)

        prompts = CUBE_PROMPTS.get(color.lower(), [f"small {color} {target_type}"])
        valid_bboxes = []
        for p in prompts:
            res = self._infer(image, p)
            if res["target"]["found"]:
                valid_bboxes.append(res["target"]["bbox_xyxy"])

        if not valid_bboxes:
            return {"target": {"found": False}}

        median_bbox = np.median(np.array(valid_bboxes), axis=0).astype(int).tolist()
        return {
            "target": {
                "found": True,
                "bbox_xyxy": median_bbox,
                "num_valid": len(valid_bboxes),
            }
        }
