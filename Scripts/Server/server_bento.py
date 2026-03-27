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
    "red": [
        "red cube in the center of the floor near the robot arm",
        "small red block located directly in front of the robot base",
        "top-down view of a red cube centered on the grey surface",
    ],
    "green": [
        "small green cube on the far right side of the floor",
        "the tiny green block away from the robot arm",
        "green square object in the right-hand area of the image",
    ],
    "blue": [
        "small blue cube on the far left side of the floor",
        "the tiny blue block away from the robot arm",
        "blue square object in the left-hand area of the image",
    ],
}

PALLET_PROMPTS = {
    "red": [
        "red plastic pallet on the floor",
        "top view of a red cargo pallet near the robot",
        "the center of the red rectangular grid structure"
    ],
    "blue": [
        "blue plastic pallet located to the left",
        "blue industrial pallet on the grey surface",
        "top-down view of a blue rectangular pallet"
    ],
    "black": [ 
        "black plastic pallet on the right side",
        "dark grey cargo pallet on the floor",
        "rectangular black grid structure"
    ]
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
        print(f"----- MODELO: {self.model_id} CARGADO -----")

    def _is_round_number_bbox(self, coords):
        """Filtro para detectar bboxes falsos que terminan en 00 o 50 (alucinación común)."""
        return all(c % 50 == 0 for c in coords)
    
    def _is_hallucination(self, coords):
        """
        Filtro de tamaño dinámico. 
        Permite hasta 600px para palés, pero ignora cosas demasiado pequeñas o gigantes.
        """
        ymin, xmin, ymax, xmax = coords
        width = xmax - xmin 
        height = ymax - ymin 
        
        if width > 600 or height > 600:
            return True 
        if width < 15 or height < 15:
            return True
        return False

    def _infer(self, image: Image.Image, instruction: str) -> dict:
        """Ejecuta una inferencia única con el VLM."""
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

    def extract_coords(self, text):
        """Extrae y filtra las coordenadas del texto de salida."""
        bracket_match = re.search(
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text
        )

        if bracket_match:
            coords = [int(x) for x in bracket_match.groups()]
            
            if coords[0] < 5 and coords[1] < 5: # Pegado a la esquina superior
                return {"target": {"found": False}}
            if self._is_round_number_bbox(coords):
                return {"target": {"found": False}}
            if self._is_hallucination(coords):
                return {"target": {"found": False}}
                
            return {"target": {"bbox_xyxy": coords, "found": True}}
        return {"target": {"found": False}}

    @bentoml.api
    async def ground_multi(self, color: str, image_b64: str, target_type: str = "cube") -> dict:
        """
        Detección múltiple optimizada para Cubos o Palés.
        """
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if target_type.lower() == "pallet":
            prompts = PALLET_PROMPTS.get(color.lower(), [f"{color} plastic pallet"])
        else:
            prompts = CUBE_PROMPTS.get(color.lower(), [f"{color} small cube"])

        valid_bboxes = []
        for prompt in prompts:
            result = self._infer(image, prompt)
            if result and result.get("target") and result["target"].get("found"):
                valid_bboxes.append(result["target"]["bbox_xyxy"])

        if len(valid_bboxes) == 0:
            return {"target": {"found": False, "num_valid": 0}}

        if len(valid_bboxes) >= 2:
            areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in valid_bboxes]
            median_area = np.median(areas)
            valid_bboxes = [b for b, a in zip(valid_bboxes, areas) 
                            if abs(a - median_area) < median_area * 1.0] # 70% margen

        if not valid_bboxes:
            return {"target": {"found": False, "num_valid": 0}}

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
    async def ground(self, instruction: str, image_b64: str) -> dict:
        """Detección simple para compatibilidad."""
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return self._infer(image, instruction)