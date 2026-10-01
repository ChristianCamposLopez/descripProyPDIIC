from pathlib import Path
from collections import Counter
import cv2
import json

class DatasetAnalyzer:
    """
    Responsable único de analizar el dataset de imágenes (Single Responsibility).
    """
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def find_images(self) -> list[Path]:
        """Busca únicamente archivos .jpg, eliminando la redundancia de múltiples extensiones."""
        return sorted(self.dataset_path.rglob("*.jpg"))

    def _classify_split(self, path: Path) -> str:
        parts = {part.lower() for part in path.parts}
        if "train" in parts:
            return "train"
        if "valid" in parts or "validation" in parts:
            return "valid"
        if "test" in parts:
            return "test"
        return "unknown"

    def summarize(self, images: list[Path]) -> dict:
        split_counts = Counter()
        class_counts = Counter()
        sizes = Counter()
        color_modes = Counter()
        errors = []

        for path in images:
            split = self._classify_split(path)
            split_counts[split] += 1
            class_name = path.parent.name if split != "test" else "test_unlabeled"
            class_counts[class_name] += 1

            # Check sizes and channels without keeping the whole image in memory longer than needed
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                errors.append({"path": str(path), "error": "No se pudo leer"})
                continue
            
            height, width = image.shape[:2]
            channels = 1 if image.ndim == 2 else image.shape[2]
            sizes[f"{width}x{height}"] += 1
            
            # Using standard naming 'by_mode'
            mode_name = "RGB" if channels == 3 else f"channels_{channels}"
            color_modes[mode_name] += 1

        return {
            "total": len(images),
            "by_split": dict(split_counts),
            "by_class": dict(class_counts),
            "by_format": {".jpg": len(images)},
            "by_size": dict(sizes),
            "by_mode": dict(color_modes),
            "errors": errors,
        }

    def save_stats(self, stats: dict, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
