
import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from tqdm import tqdm

from dataset.dataset import (
    ChangeDetectionDataset,
)

from models.changeformer import (
    ChangeFormer,
)

from utils.metrics import (
    MetricAccumulator,
)

from torch.utils.data import DataLoader


@torch.no_grad()

def predict_tta(
    model,
    pre,
    post,
):

    # Original
    p1 = model(
        pre,
        post,
    )

    # Horizontal flip
    pre_h = torch.flip(
        pre,
        dims=[3],
    )

    post_h = torch.flip(
        post,
        dims=[3],
    )

    p2 = model(
        pre_h,
        post_h,
    )

    p2 = torch.flip(
        p2,
        dims=[3],
    )

    # Vertical flip
    pre_v = torch.flip(
        pre,
        dims=[2],
    )

    post_v = torch.flip(
        post,
        dims=[2],
    )

    p3 = model(
        pre_v,
        post_v,
    )

    p3 = torch.flip(
        p3,
        dims=[2],
    )

    pred = (
        p1 + p2 + p3
    ) / 3.0

    return pred


def save_visualization(
    pre,
    post,
    target,
    pred,
    save_path,
):

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(20, 5),
    )

    pre = pre.permute(
        1,
        2,
        0,
    ).cpu().numpy()

    post = post.squeeze().cpu().numpy()

    target = target.squeeze().cpu().numpy()

    pred = pred.squeeze().cpu().numpy()

    ax[0].imshow(pre)
    ax[0].set_title("EO")

    ax[1].imshow(post, cmap="gray")
    ax[1].set_title("SAR")

    ax[2].imshow(target, cmap="gray")
    ax[2].set_title("GT")

    ax[3].imshow(pred, cmap="gray")
    ax[3].set_title("Prediction")

    for a in ax:

        a.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight",
    )

    plt.close()


@torch.no_grad()

def evaluate(
    model,
    loader,
    device,
    threshold,
    save_dir,
):

    model.eval()

    accumulator = MetricAccumulator()

    vis_dir = os.path.join(
        save_dir,
        "visualizations",
    )

    os.makedirs(
        vis_dir,
        exist_ok=True,
    )

    for idx, batch in enumerate(
        tqdm(loader)
    ):

        pre = batch["pre"].to(device)

        post = batch["post"].to(device)

        target = batch["target"].to(device)

        pred = predict_tta(
            model,
            pre,
            post,
        )

        accumulator.update(
            pred,
            target,
            threshold=threshold,
        )

        if idx < 10:

            binary_pred = (
                torch.sigmoid(pred)
                > threshold
            ).float()

            save_visualization(

                pre[0].cpu(),

                post[0].cpu(),

                target[0].cpu(),

                binary_pred[0].cpu(),

                os.path.join(
                    vis_dir,
                    f"sample_{idx}.png",
                ),
            )

    metrics = accumulator.print_summary(
        split="test"
    )

    # Save metrics
    metrics_path = os.path.join(
        save_dir,
        "metrics.txt",
    )

    with open(metrics_path, "w") as f:

        f.write(
            "── TEST Metrics ────────────────────────\n"
        )

        f.write(
            f"IoU: {metrics['iou']}\n"
        )

        f.write(
            f"Precision: {metrics['precision']}\n"
        )

        f.write(
            f"Recall: {metrics['recall']}\n"
        )

        f.write(
            f"F1: {metrics['f1']}\n"
        )

        cm = metrics["confusion_matrix"]

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            f"TP: {cm['tp']}\n"
        )

        f.write(
            f"FP: {cm['fp']}\n"
        )

        f.write(
            f"FN: {cm['fn']}\n"
        )

        f.write(
            f"TN: {cm['tn']}\n"
        )

    print(
        f"\nMetrics saved -> {metrics_path}"
    )

    print(
        f"10 visualisations saved -> {vis_dir}"
    )

    return metrics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="eval_results",
    )

    args = parser.parse_args()

    with open(args.config) as f:

        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # Dataset
    test_ds = ChangeDetectionDataset(

        root_dir=cfg["data"]["test_dir"],

        split="test",

        image_size=cfg["data"]["image_size"],
    )

    test_loader = DataLoader(

        test_ds,

        batch_size=1,

        shuffle=False,

        num_workers=2,
    )

    # Model
    model = ChangeFormer().to(device)

    checkpoint = torch.load(
        args.weights,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"\nLoaded weights from {args.weights}"
    )

    evaluate(

        model,

        test_loader,

        device,

        args.threshold,

        args.save_dir,
    )


if __name__ == "__main__":

    main()
