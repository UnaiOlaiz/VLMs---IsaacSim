# server_bento.py

import base64
import io
import json
import re

import bentoml
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


@bentoml.service(name="VLM_Service_Isaac", resources={"gpu": 1})
class VLMServiceIsaac:
    def __init__(self):
        self.model_id = MODEL_ID

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
            device_map="auto",
        ).eval()

        print(f"----- MODEL LOADED: {self.model_id} -----")

    @bentoml.api
    async def ground(
        self,
        instruction: str,
        image_b64: str,
        camera_id: str = "",
        robot_id: str = "",
    ) -> dict:
        """
        Ground one instruction in one image.

        Args:
            instruction: text prompt for the target, e.g. "red platform"
            image_b64: base64-encoded RGB image
            camera_id: optional camera identifier, useful for multi-camera setups
            robot_id: optional robot identifier, useful for multi-robot setups

        Returns:
            {
                "target": {
                    "found": bool,
                    "bbox_xyxy": [ymin, xmin, ymax, xmax]   # if found
                },
                "camera_id": "...",
                "robot_id": "...",
                "raw_text": "..."
            }
        """
        # -------------------------------------------------
        # Decode image
        # -------------------------------------------------
        try:
            img_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return {
                "target": {"found": False},
                "camera_id": camera_id,
                "robot_id": robot_id,
                "error": f"image_decode_error: {str(e)}",
            }

        # -------------------------------------------------
        # Strict grounding prompt
        # -------------------------------------------------
        prompt = (
            f"Locate exactly one instance of: {instruction}.\n"
            "Return only valid JSON.\n"
            'If found, return: {"target": {"found": true, "bbox_xyxy": [ymin, xmin, ymax, xmax]}}\n'
            'If not found, return: {"target": {"found": false}}\n'
            "Use normalized integer coordinates from 0 to 1000.\n"
            "Do not return explanations, markdown, comments, or extra text."
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

        # -------------------------------------------------
        # Generate model output
        # -------------------------------------------------
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, _ = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )

            trimmed_ids = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            out_text = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
            )[0]

        except Exception as e:
            return {
                "target": {"found": False},
                "camera_id": camera_id,
                "robot_id": robot_id,
                "error": f"generation_error: {str(e)}",
            }

        # -------------------------------------------------
        # Parse result
        # -------------------------------------------------
        parsed = self.extract_coords(out_text)
        parsed["camera_id"] = camera_id
        parsed["robot_id"] = robot_id
        parsed["raw_text"] = out_text

        return parsed

    def extract_coords(self, text: str) -> dict:
        """
        Try several parsing strategies:
        1) strict JSON parse
        2) extract JSON object substring
        3) fallback regex for bbox list [ymin, xmin, ymax, xmax]
        """

        # -------------------------------------------------
        # 1) Strict JSON parse
        # -------------------------------------------------
        try:
            parsed = json.loads(text)
            if self._valid_target_dict(parsed):
                return self._sanitize_target_dict(parsed)
        except Exception:
            pass

        # -------------------------------------------------
        # 2) Extract first JSON-like object
        # -------------------------------------------------
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if self._valid_target_dict(parsed):
                    return self._sanitize_target_dict(parsed)
            except Exception:
                pass

        # -------------------------------------------------
        # 3) Fallback regex for bracketed bbox
        # -------------------------------------------------
        bracket_match = re.search(
            r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
            text,
        )
        if bracket_match:
            coords = [int(x) for x in bracket_match.groups()]
            coords = [max(0, min(1000, c)) for c in coords]

            ymin, xmin, ymax, xmax = coords

            if ymax < ymin:
                ymin, ymax = ymax, ymin
            if xmax < xmin:
                xmin, xmax = xmax, xmin

            return {
                "target": {
                    "found": True,
                    "bbox_xyxy": [ymin, xmin, ymax, xmax],
                }
            }

        return {"target": {"found": False}}

    def _valid_target_dict(self, parsed: dict) -> bool:
        if not isinstance(parsed, dict):
            return False
        if "target" not in parsed:
            return False
        if not isinstance(parsed["target"], dict):
            return False
        if "found" not in parsed["target"]:
            return False

        found = parsed["target"]["found"]

        if not isinstance(found, bool):
            return False

        if found is True:
            bbox = parsed["target"].get("bbox_xyxy", None)
            if not isinstance(bbox, list) or len(bbox) != 4:
                return False
            if not all(isinstance(x, int) for x in bbox):
                return False

        return True

    def _sanitize_target_dict(self, parsed: dict) -> dict:
        """
        Clamp bbox values into [0, 1000] and ensure ordering:
        [ymin, xmin, ymax, xmax]
        """
        if not parsed["target"]["found"]:
            return {"target": {"found": False}}

        bbox = parsed["target"]["bbox_xyxy"]
        bbox = [max(0, min(1000, int(v))) for v in bbox]

        ymin, xmin, ymax, xmax = bbox

        if ymax < ymin:
            ymin, ymax = ymax, ymin
        if xmax < xmin:
            xmin, xmax = xmax, xmin

        return {
            "target": {
                "found": True,
                "bbox_xyxy": [ymin, xmin, ymax, xmax],
            }
        }
