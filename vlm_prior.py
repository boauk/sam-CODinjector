import json
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


@dataclass
class VLMImagePrior:
    confidence: float
    grounding_box: List[float]
    text: str

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "VLMImagePrior":
        return cls(
            confidence=_clamp_float(raw.get("confidence", 0.5), 0.0, 1.0),
            grounding_box=normalize_grounding_box(
                raw.get("grounding_box", [0.0, 0.0, 1.0, 1.0])
            ),
            text=str(raw.get("text", "")).strip(),
        )

    @classmethod
    def from_json(cls, raw: Union[str, Dict[str, Any]]) -> "VLMImagePrior":
        if isinstance(raw, str):
            raw = json.loads(raw)
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class RuntimeVLMImagePrior:
    serializable: VLMImagePrior
    text_features: torch.Tensor
    prior_mask: torch.Tensor

    @property
    def confidence_tensor(self) -> torch.Tensor:
        return torch.tensor([self.serializable.confidence], dtype=torch.float32)


class PriorCache:
    def __init__(self, cache_dir: Union[str, Path], mask_size: Tuple[int, int] = (1024, 1024)):
        self.cache_dir = Path(cache_dir)
        self.mask_size = mask_size
        self.hits = 0
        self.misses = 0
        self.saves = 0

    def cache_key(self, image_path: Union[str, Path], target_text: Optional[str] = None) -> str:
        image_path = str(image_path)
        stem = Path(image_path).stem or "image"
        fingerprint_source = f"{Path(image_path).as_posix()}|{target_text or ''}"
        digest = hashlib.sha1(fingerprint_source.encode("utf-8")).hexdigest()[:12]
        return f"{stem}_{digest}"

    def json_path(self, image_path: Union[str, Path], target_text: Optional[str] = None) -> Path:
        return self.cache_dir / f"{self.cache_key(image_path, target_text)}.json"

    def tensor_path(self, image_path: Union[str, Path], target_text: Optional[str] = None) -> Path:
        return self.cache_dir / f"{self.cache_key(image_path, target_text)}.pt"

    def load(
        self,
        image_path: Union[str, Path],
        target_text: Optional[str] = None,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Optional[RuntimeVLMImagePrior]:
        json_path = self.json_path(image_path, target_text)
        tensor_path = self.tensor_path(image_path, target_text)
        if not json_path.exists() or not tensor_path.exists():
            self.misses += 1
            return None

        self.hits += 1

        with json_path.open("r", encoding="utf-8") as f:
            serializable = VLMImagePrior.from_dict(json.load(f))
        tensors = torch.load(tensor_path, map_location=map_location)
        text_features = tensors["text_features"]
        prior_mask = torch.from_numpy(
            box_to_mask(
                serializable.grounding_box,
                height=int(self.mask_size[0]),
                width=int(self.mask_size[1]),
            )
        ).unsqueeze(0)
        return RuntimeVLMImagePrior(
            serializable=serializable,
            text_features=text_features,
            prior_mask=prior_mask,
        )

    def save(
        self,
        image_path: Union[str, Path],
        prior: RuntimeVLMImagePrior,
        target_text: Optional[str] = None,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self.json_path(image_path, target_text).open("w", encoding="utf-8") as f:
            json.dump(prior.serializable.to_dict(), f, ensure_ascii=False, indent=2)
        torch.save(
            {"text_features": prior.text_features.detach().cpu()},
            self.tensor_path(image_path, target_text),
        )
        self.saves += 1

    def summary(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "saves": self.saves}


class LazyGroundingDinoRuntimePriorGenerator:
    def __init__(
        self,
        blip_model_name: str = "Salesforce/blip-image-captioning-large",
        dino_model_name: str = "IDEA-Research/grounding-dino-base",
        device: Optional[Union[str, torch.device]] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        mask_size: Tuple[int, int] = (1024, 1024),
        generator_factory=None,
    ):
        self._generator = None
        self._generator_factory = generator_factory
        self._generator_kwargs = {
            "blip_model_name": blip_model_name,
            "dino_model_name": dino_model_name,
            "device": device,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "mask_size": mask_size,
        }

    @property
    def loaded(self) -> bool:
        return self._generator is not None

    def _get_generator(self):
        if self._generator is None:
            if self._generator_factory is not None:
                self._generator = self._generator_factory()
            else:
                self._generator = GroundingDinoRuntimePriorGenerator(
                    **self._generator_kwargs
                )
        return self._generator

    def generate(self, image, target_text: Optional[str] = None):
        return self._get_generator().generate(image, target_text=target_text)


_GENERIC_DATASET_PATH_PARTS = {
    "gt",
    "gts",
    "image",
    "images",
    "img",
    "imgs",
    "mask",
    "masks",
    "test",
    "train",
    "training",
    "val",
    "valid",
    "validation",
}


def resolve_prior_cache_dir(
    prior_cache_dir: Optional[str],
    dataset_path: Union[str, Path],
    default_base_dir: Union[str, Path] = Path("work_dir") / "vlm_prior_cache",
) -> Optional[str]:
    if prior_cache_dir is not None:
        return prior_cache_dir

    dataset_name = _dataset_cache_name(dataset_path)
    return str(Path(default_base_dir) / dataset_name)


def _dataset_cache_name(dataset_path: Union[str, Path]) -> str:
    path = Path(str(dataset_path).strip())
    if path.suffix:
        path = path.parent
    while path.name and path.name.lower() in _GENERIC_DATASET_PATH_PARTS:
        parent = path.parent
        if parent == path:
            break
        path = parent

    name = path.name or "dataset"
    name = re.sub(r'[<>:"/\\|?*\s]+', "_", name).strip("._")
    return name or "dataset"


def load_or_generate_prior(
    prior_generator,
    image,
    image_path: Union[str, Path],
    target_text: Optional[str] = None,
    prior_cache: Optional[PriorCache] = None,
) -> RuntimeVLMImagePrior:
    if prior_cache is not None:
        cached = prior_cache.load(image_path, target_text=target_text)
        if cached is not None:
            return cached

    prior = prior_generator.generate(image, target_text=target_text)
    if prior_cache is not None:
        prior_cache.save(image_path, prior, target_text=target_text)
    return prior


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = low
    return min(high, max(low, value))


def normalize_grounding_box(grounding_box: Any) -> List[float]:
    if not isinstance(grounding_box, (list, tuple)) or len(grounding_box) != 4:
        return [0.0, 0.0, 1.0, 1.0]

    x0, y0, x1, y1 = [_clamp_float(v, 0.0, 1.0) for v in grounding_box]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def box_to_mask(grounding_box: List[float], height: int, width: int) -> np.ndarray:
    x0, y0, x1, y1 = normalize_grounding_box(grounding_box)
    left = int(round(x0 * width))
    top = int(round(y0 * height))
    right = int(round(x1 * width))
    bottom = int(round(y1 * height))

    mask = np.zeros((1, height, width), dtype=np.float32)
    if right > left and bottom > top:
        mask[:, top:bottom, left:right] = 1.0
    return mask


def image_to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        image = image.detach().cpu()
        if image.dim() == 4:
            if image.shape[0] != 1:
                raise ValueError("image_to_pil only accepts a single image batch")
            image = image[0]
        if image.dim() == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("image_to_pil only accepts a single image batch")
            image = image[0]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            return Image.fromarray(image).convert("RGB")
        return Image.fromarray(image).convert("RGB")
    raise ValueError("Unsupported image type for image_to_pil")


class GroundingDinoRuntimePriorGenerator:
    DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-large"
    DEFAULT_DINO_MODEL = "IDEA-Research/grounding-dino-base"

    def __init__(
        self,
        blip_model_name: str = DEFAULT_BLIP_MODEL,
        dino_model_name: str = DEFAULT_DINO_MODEL,
        device: Optional[Union[str, torch.device]] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        mask_size: Tuple[int, int] = (1024, 1024),
    ):
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                BlipForConditionalGeneration,
                BlipProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "GroundingDinoRuntimePriorGenerator requires transformers."
            ) from exc

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.mask_size = mask_size
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            blip_model_name
        ).to(self.device)
        self.blip_model.eval()
        self.dino_processor = AutoProcessor.from_pretrained(dino_model_name)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model_name
        ).to(self.device)
        self.dino_model.eval()

    @torch.no_grad()
    def generate(self, image, target_text: Optional[str] = None) -> RuntimeVLMImagePrior:
        text = target_text or self.generate_blip_text(image)
        text_prompt = self._format_text_prompt(text)
        inputs = self.dino_processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)
        outputs = self.dino_model(**inputs, output_hidden_states=True)
        text_features = self.extract_text_features(outputs)

        image_size = self.get_image_size(image)
        result = self.post_process_grounding_dino(
            outputs=outputs,
            inputs=inputs,
            image_size=image_size,
        )
        return self.prior_from_detection_result(
            result=result,
            image_size=image_size,
            text=text,
            text_features=text_features,
            mask_size=self.mask_size,
        )

    @torch.no_grad()
    def generate_blip_text(self, image) -> str:
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        outputs = self.blip_model.generate(**inputs)
        return self.blip_processor.decode(outputs[0], skip_special_tokens=True).strip()

    def post_process_grounding_dino(self, outputs, inputs, image_size):
        post_process = getattr(
            self.dino_processor, "post_process_grounded_object_detection", None
        )
        if post_process is None:
            raise AttributeError(
                "Grounding DINO processor does not expose "
                "post_process_grounded_object_detection."
            )
        target_size = (image_size[1], image_size[0])
        try:
            results = post_process(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[target_size],
            )
        except TypeError:
            results = post_process(
                outputs,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[target_size],
            )
        return results[0]

    @staticmethod
    def prior_from_detection_result(
        result: Dict[str, Any],
        image_size: Tuple[int, int],
        text: str,
        text_features: torch.Tensor,
        mask_size: Tuple[int, int],
    ) -> RuntimeVLMImagePrior:
        scores = torch.as_tensor(result.get("scores", []), dtype=torch.float32)
        if scores.numel() == 0:
            serializable = VLMImagePrior(
                confidence=0.0,
                grounding_box=[0.0, 0.0, 1.0, 1.0],
                text=text,
            )
        else:
            best_idx = int(torch.argmax(scores).item())
            boxes = torch.as_tensor(result.get("boxes"), dtype=torch.float32)
            box = boxes[best_idx].detach().cpu().tolist()
            width, height = float(image_size[0]), float(image_size[1])
            serializable = VLMImagePrior(
                confidence=_clamp_float(scores.flatten()[best_idx].item(), 0.0, 1.0),
                grounding_box=normalize_grounding_box(
                    [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
                ),
                text=GroundingDinoRuntimePriorGenerator._label_from_result(
                    result, best_idx, text
                ),
            )

        prior_mask = torch.from_numpy(
            box_to_mask(
                serializable.grounding_box,
                height=int(mask_size[0]),
                width=int(mask_size[1]),
            )
        ).unsqueeze(0)
        return RuntimeVLMImagePrior(
            serializable=serializable,
            text_features=text_features.detach(),
            prior_mask=prior_mask,
        )

    @staticmethod
    def extract_text_features(outputs) -> torch.Tensor:
        for name in (
            "encoder_last_hidden_state",
            "last_hidden_state",
            "text_embeds",
            "text_features",
        ):
            value = getattr(outputs, name, None)
            if torch.is_tensor(value):
                return value
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states and torch.is_tensor(hidden_states[-1]):
            return hidden_states[-1]
        raise AttributeError("Could not find text features in Grounding DINO output.")

    @staticmethod
    def get_image_size(image) -> Tuple[int, int]:
        if hasattr(image, "size") and not callable(image.size):
            return int(image.size[0]), int(image.size[1])
        if torch.is_tensor(image):
            return int(image.shape[-1]), int(image.shape[-2])
        if isinstance(image, np.ndarray):
            return int(image.shape[1]), int(image.shape[0])
        raise ValueError("image must be a PIL image, numpy array, or torch tensor")

    @staticmethod
    def _format_text_prompt(text: str) -> str:
        text = text.strip().lower()
        return text if text.endswith(".") else f"{text}."

    @staticmethod
    def _label_from_result(result: Dict[str, Any], index: int, fallback: str) -> str:
        labels = result.get("text_labels", result.get("labels", []))
        if isinstance(labels, torch.Tensor) or index >= len(labels):
            return fallback
        label = str(labels[index]).strip()
        return label if label and not label.isdigit() else fallback
