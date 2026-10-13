import argparse, os, yaml
from PIL import Image
import numpy as np
from intellisegment.pipeline import IntelliSegment, save_outputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="outputs")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    image = np.asarray(Image.open(args.image).convert("RGB"))
    model = IntelliSegment(config)
    outputs = model.run(image)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    save_outputs(image, outputs, args.output, stem)
    print(f"Kept {len(outputs)} verified detections. Outputs: {args.output}")

if __name__ == "__main__":
    main()
