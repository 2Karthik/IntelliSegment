from pathlib import Path
import numpy as np
from PIL import Image

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))

def load_mask(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

def matching_files(directory, extensions=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    directory = Path(directory)
    return {p.stem: p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions}
