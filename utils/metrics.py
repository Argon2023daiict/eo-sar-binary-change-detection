
import torch


def compute_metrics(
    preds,
    targets,
    threshold=0.8,
):

    preds = torch.sigmoid(
        preds
    )

    preds = (
        preds > threshold
    ).float()

    targets = targets.float()

    preds = preds.view(-1)

    targets = targets.view(-1)

    tp = (
        (preds == 1)
        &
        (targets == 1)
    ).sum().item()

    fp = (
        (preds == 1)
        &
        (targets == 0)
    ).sum().item()

    fn = (
        (preds == 0)
        &
        (targets == 1)
    ).sum().item()

    tn = (
        (preds == 0)
        &
        (targets == 0)
    ).sum().item()

    eps = 1e-7

    precision = tp / (
        tp + fp + eps
    )

    recall = tp / (
        tp + fn + eps
    )

    f1 = (
        2
        * precision
        * recall
    ) / (
        precision + recall + eps
    )

    iou = tp / (
        tp + fp + fn + eps
    )

    return {

        "iou": iou,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "tp": tp,

        "fp": fp,

        "fn": fn,

        "tn": tn,
    }


class MetricAccumulator:

    def __init__(self):

        self.reset()

    def reset(self):

        self.tp = 0

        self.fp = 0

        self.fn = 0

        self.tn = 0

    def update(
        self,
        preds,
        targets,
        threshold=0.8,
    ):

        m = compute_metrics(
            preds,
            targets,
            threshold,
        )

        self.tp += m["tp"]

        self.fp += m["fp"]

        self.fn += m["fn"]

        self.tn += m["tn"]

    def compute(self):

        eps = 1e-7

        precision = self.tp / (
            self.tp + self.fp + eps
        )

        recall = self.tp / (
            self.tp + self.fn + eps
        )

        f1 = (
            2
            * precision
            * recall
        ) / (
            precision + recall + eps
        )

        iou = self.tp / (
            self.tp
            + self.fp
            + self.fn
            + eps
        )

        return {

            "iou": round(iou, 4),

            "precision": round(
                precision,
                4,
            ),

            "recall": round(
                recall,
                4,
            ),

            "f1": round(
                f1,
                4,
            ),

            "confusion_matrix": {

                "tp": int(self.tp),

                "fp": int(self.fp),

                "fn": int(self.fn),

                "tn": int(self.tn),
            }
        }

    def print_summary(
        self,
        split="val",
    ):

        m = self.compute()

        print(
            f"\n── {split.upper()} Metrics ────────────────────────"
        )

        print(
            f"  IoU:       {m['iou']:.4f}"
        )

        print(
            f"  Precision: {m['precision']:.4f}"
        )

        print(
            f"  Recall:    {m['recall']:.4f}"
        )

        print(
            f"  F1:        {m['f1']:.4f}"
        )

        cm = m["confusion_matrix"]

        print(
            "  Confusion Matrix:"
        )

        print(
            f"    TP={cm['tp']:>8d}  FP={cm['fp']:>8d}"
        )

        print(
            f"    FN={cm['fn']:>8d}  TN={cm['tn']:>8d}"
        )

        print(
            "──────────────────────────────────────────"
        )

        return m
