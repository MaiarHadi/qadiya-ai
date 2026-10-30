
import cv2
import numpy as np

from app.services.image_processing import ImageProcessingError

MAX_OUTPUT_PIXELS = 16_000_000
#size max1


def denoise(img: np.ndarray, strength: int) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
   
    h, w = img.shape[:2]
    if (w * factor) * (h * factor) > MAX_OUTPUT_PIXELS:
        raise ImageProcessingError("   vary big  ")
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


def clahe(img: np.ndarray) -> np.ndarray:
   
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def brightness_contrast(img: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
    
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def sharpen(img: np.ndarray, amount: float) -> np.ndarray:
   
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def to_grayscale(img: np.ndarray) -> np.ndarray:
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def enhance_image(
    img: np.ndarray,
    *,
    brightness: int = 0,
    contrast: float = 1.0,
    grayscale: bool = False,
    sharpen_amount: float = 0.0,
    denoise_strength: int = 0,
    use_clahe: bool = False,
    upscale_factor: int = 1,
) -> np.ndarray:
    
    out = img.copy() 

    if denoise_strength > 0:
        out = denoise(out, denoise_strength)

    
    if upscale_factor > 1:
        out = upscale(out, upscale_factor)

    
    if use_clahe:
        out = clahe(out)

    if brightness != 0 or contrast != 1.0:
        out = brightness_contrast(out, brightness, contrast)

    
    if sharpen_amount > 0:
        out = sharpen(out, sharpen_amount)

   
    if grayscale:
        out = to_grayscale(out)

    return out