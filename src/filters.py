from abc import ABC, abstractmethod
import cv2
import numpy as np

class ImageFilter(ABC):
    """
    Base class for all image filters, following the Open/Closed Principle.
    """
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        pass


class CLAHEEqualization(ImageFilter):
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced_l = clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


class GlobalEqualization(ImageFilter):
    """Ecualización clásica del histograma en el canal L para preservar color."""
    def apply(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        equalized_l = cv2.equalizeHist(l_channel)
        merged = cv2.merge((equalized_l, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


class MedianFilter(ImageFilter):
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(image, self.kernel_size)


class BilateralFilter(ImageFilter):
    def __init__(self, d: int = 5, sigma_color: float = 35.0, sigma_space: float = 35.0):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(image, self.d, self.sigma_color, self.sigma_space)


class GaussianBlurFilter(ImageFilter):
    def __init__(self, kernel_size: tuple[int, int] = (5, 5), sigma_x: float = 0):
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, self.kernel_size, self.sigma_x)


class UnsharpMaskFilter(ImageFilter):
    def __init__(self, sigma_x: float = 1.2, amount: float = 1.5):
        self.sigma_x = sigma_x
        self.amount = amount

    def apply(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.sigma_x)
        # amount * image - (amount - 1) * blurred
        return cv2.addWeighted(image, self.amount, blurred, -(self.amount - 1.0), 0)


class ExGLeafSegmenter(ImageFilter):
    """
    Extrae la máscara de la hoja usando el índice Excess Green (ExG)
    y atenúa el fondo para resaltarla.
    """
    def apply(self, image: np.ndarray) -> np.ndarray:
        # Convert to float for calculation
        img_float = image.astype(np.float32)
        b, g, r = cv2.split(img_float)
        
        # Calculate ExG = 2*G - R - B
        exg = 2.0 * g - r - b
        
        # Normalize to 0-255 uint8
        exg_norm = cv2.normalize(exg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        exg_u8 = exg_norm.astype(np.uint8)
        
        # Otsu's thresholding
        _, mask = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological opening to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Create output: Leaf regions stay original, background is pure black
        leaf_region = cv2.bitwise_and(image, image, mask=mask_clean)
        
        return leaf_region


class Pipeline(ImageFilter):
    """
    Ejecuta una secuencia de filtros siguiendo el Patrón Composite/Pipeline.
    """
    def __init__(self, filters: list[ImageFilter]):
        self.filters = filters

    def apply(self, image: np.ndarray) -> np.ndarray:
        result = image
        for f in self.filters:
            result = f.apply(result)
        return result
