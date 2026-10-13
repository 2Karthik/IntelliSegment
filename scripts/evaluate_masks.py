import argparse
import pandas as pd
from intellisegment.data import load_mask, matching_files
from intellisegment.metrics import evaluate_pair

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--dataset", choices=["ecssd", "pascal_voc"], required=True)
    ap.add_argument("--output", default="metrics.csv")
    args = ap.parse_args()
    pred_files, gt_files = matching_files(args.pred_dir), matching_files(args.gt_dir)
    common = sorted(set(pred_files) & set(gt_files))
    if not common:
        raise SystemExit("No matching prediction/ground-truth filenames found.")
    rows = []
    for stem in common:
        pred, gt = load_mask(pred_files[stem]), load_mask(gt_files[stem])
        if pred.shape != gt.shape:
            from PIL import Image
            gt_img = Image.open(gt_files[stem]).convert("L").resize((pred.shape[1], pred.shape[0]))
            import numpy as np
            gt = np.asarray(gt_img, dtype=np.float32) / 255.0
        row = {"image": stem, **evaluate_pair(pred, gt, args.dataset)}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(df.mean(numeric_only=True).to_string())
    print(f"Saved per-image results to {args.output}")

if __name__ == "__main__":
    main()
