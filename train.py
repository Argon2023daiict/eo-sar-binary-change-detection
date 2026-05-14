
import os
import yaml
import random
import argparse
import numpy as np

import torch

from tqdm import tqdm

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
)

from torch.optim import AdamW

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
)

from torch.amp import (
    autocast,
    GradScaler,
)

from dataset.dataset import (
    ChangeDetectionDataset,
)

from models.changeformer import (
    ChangeFormer,
)

from utils.losses import (
    BCE_Dice_Loss,
)

from utils.metrics import (
    MetricAccumulator,
)


def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
):

    model.train()

    accumulator = MetricAccumulator()

    running_loss = 0.0

    pbar = tqdm(loader)

    for batch in pbar:

        pre = batch["pre"].to(device)

        post = batch["post"].to(device)

        target = batch["target"].to(device)

        optimizer.zero_grad()

        with autocast("cuda"):

            pred = model(
                pre,
                post,
            )

            loss, bce_val, dice_val = criterion(
                pred,
                target,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        accumulator.update(
            pred.detach(),
            target.detach(),
        )

        pbar.set_postfix({

            "loss": f"{loss.item():.4f}",

            "bce": f"{bce_val:.4f}",

            "dice": f"{dice_val:.4f}",
        })

    metrics = accumulator.compute()

    return (
        running_loss / len(loader),
        metrics,
    )


@torch.no_grad()

def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    accumulator = MetricAccumulator()

    running_loss = 0.0

    for batch in tqdm(loader):

        pre = batch["pre"].to(device)

        post = batch["post"].to(device)

        target = batch["target"].to(device)

        with autocast("cuda"):

            pred = model(
                pre,
                post,
            )

            loss, _, _ = criterion(
                pred,
                target,
            )

        running_loss += loss.item()

        accumulator.update(
            pred,
            target,
        )

    metrics = accumulator.compute()

    return (
        running_loss / len(loader),
        metrics,
    )


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_f1,
    path,
):

    torch.save({

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "best_f1": best_f1,

    }, path)

    print(
        f"  Checkpoint saved → {path}"
    )


def main(config_path):

    with open(config_path) as f:

        cfg = yaml.safe_load(f)

    seed_everything(
        cfg["training"]["seed"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    train_ds = ChangeDetectionDataset(

        root_dir=cfg["data"]["train_dir"],

        split="train",

        image_size=cfg["data"]["image_size"],
    )

    val_ds = ChangeDetectionDataset(

        root_dir=cfg["data"]["val_dir"],

        split="val",

        image_size=cfg["data"]["image_size"],
    )

    print(
        "\nBuilding weighted sampler..."
    )

    weights = [3.0] * len(train_ds)

    sampler = WeightedRandomSampler(

        weights,

        num_samples=len(weights),

        replacement=True,
    )

    train_loader = DataLoader(

        train_ds,

        batch_size=cfg["training"]["batch_size"],

        sampler=sampler,

        num_workers=cfg["data"]["num_workers"],

        pin_memory=True,
    )

    val_loader = DataLoader(

        val_ds,

        batch_size=cfg["training"]["batch_size"],

        shuffle=False,

        num_workers=cfg["data"]["num_workers"],

        pin_memory=True,
    )

    model = ChangeFormer().to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: {total_params/1e6:.2f}M"
    )

    criterion = BCE_Dice_Loss()

    optimizer = AdamW(

        model.parameters(),

        lr=cfg["training"]["learning_rate"],

        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = CosineAnnealingLR(

        optimizer,

        T_max=cfg["training"]["epochs"],
    )

    scaler = GradScaler("cuda")

    os.makedirs(
        cfg["checkpoint"]["save_dir"],
        exist_ok=True,
    )

    best_f1 = 0.0

    patience = 0

    for epoch in range(
        cfg["training"]["epochs"]
    ):

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch+1}/{cfg['training']['epochs']}  lr={current_lr:.2e}"
        )

        train_loss, train_metrics = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            scaler,

            device,
        )

        val_loss, val_metrics = validate(

            model,

            val_loader,

            criterion,

            device,
        )

        scheduler.step()

        print(
            f"  train loss={train_loss:.4f} "
            f"F1={train_metrics['f1']:.4f} "
            f"IoU={train_metrics['iou']:.4f}"
        )

        print(
            f"  val   loss={val_loss:.4f} "
            f"F1={val_metrics['f1']:.4f} "
            f"IoU={val_metrics['iou']:.4f}"
        )

        latest_path = os.path.join(

            cfg["checkpoint"]["save_dir"],

            "latest_model.pth",
        )

        save_checkpoint(

            model,

            optimizer,

            epoch,

            best_f1,

            latest_path,
        )

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

            best_path = os.path.join(

                cfg["checkpoint"]["save_dir"],

                "best_model.pth",
            )

            save_checkpoint(

                model,

                optimizer,

                epoch,

                best_f1,

                best_path,
            )

            patience = 0

        else:

            patience += 1

        if patience >= cfg["training"]["early_stopping_patience"]:

            print(
                "\nEarly stopping triggered."
            )

            break

    print(
        f"\nTraining complete. Best val F1 = {best_f1:.4f}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--config",

        type=str,

        required=True,
    )

    args = parser.parse_args()

    main(args.config)
