
import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0,
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets,
    ):

        probs = torch.sigmoid(
            logits
        )

        probs = probs.contiguous().view(-1)

        targets = targets.contiguous().view(-1)

        intersection = (
            probs * targets
        ).sum()

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probs.sum()
            + targets.sum()
            + self.smooth
        )

        return 1.0 - dice


class BCE_Dice_Loss(nn.Module):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5,
    ):

        super().__init__()

        self.bce_weight = bce_weight

        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(
        self,
        logits,
        targets,
    ):

        bce_loss = self.bce(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        total_loss = (

            self.bce_weight
            * bce_loss

            +

            self.dice_weight
            * dice_loss
        )

        return (

            total_loss,

            bce_loss.item(),

            dice_loss.item(),
        )
