import argparse
from metrics.pascal_metrics import evaluate_pascal
from metrics.ecssd_metrics import evaluate_ecssd

def main():
    parser = argparse.ArgumentParser(description="IntelliSegment Metric Evaluation Runner")
    parser.add_argument("--pascal_pred", type=str, default=None, help="Path to predicted masks directory for PASCAL")
    parser.add_argument("--pascal_gt", type=str, default=None, help="Path to ground truth masks directory for PASCAL")
    parser.add_argument("--ecssd_pred", type=str, default=None, help="Path to predicted masks directory for ECSSD")
    parser.add_argument("--ecssd_gt", type=str, default=None, help="Path to ground truth masks directory for ECSSD")
    
    args = parser.parse_args()

    print("=" * 60)
    print("         INTELLISEGMENT EVALUATION REPORT          ")
    print("=" * 60)

    # 1. Evaluate PASCAL Dataset
    if args.pascal_pred and args.pascal_gt:
        print("\n--> Evaluating Multi-Object Metrics on PASCAL VOC...")
        miou, mdice, f_score = evaluate_pascal(args.pascal_pred, args.pascal_gt)
        print(f"    - Mean IoU (mIoU) : {miou * 100:.2f}%")
        print(f"    - Mean Dice (mDice): {mdice * 100:.2f}%")
        print(f"    - F-Score (F_beta) : {f_score:.4f}")

    # 2. Evaluate ECSSD Dataset
    if args.ecssd_pred and args.ecssd_gt:
        print("\n--> Evaluating Single-Object Generalization Metrics on ECSSD...")
        s_alpha, e_phi, mae = evaluate_ecssd(args.ecssd_pred, args.ecssd_gt)
        print(f"    - Structure Measure (S_alpha): {s_alpha:.4f}")
        print(f"    - Enhanced Alignment (E_phi) : {e_phi:.4f}")
        print(f"    - Mean Absolute Error (MAE)  : {mae:.4f}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()