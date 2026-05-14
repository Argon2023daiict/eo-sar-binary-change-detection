
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBNReLU(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.block(x)


class FusionBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
    ):

        super().__init__()

        self.block = nn.Sequential(

            ConvBNReLU(
                in_ch,
                out_ch,
            ),

            ConvBNReLU(
                out_ch,
                out_ch,
            ),
        )

    def forward(
        self,
        f1,
        f2,
    ):

        x = torch.cat(
            [f1, f2],
            dim=1,
        )

        return self.block(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
    ):

        super().__init__()

        self.conv1 = ConvBNReLU(
            in_ch + skip_ch,
            out_ch,
        )

        self.conv2 = ConvBNReLU(
            out_ch,
            out_ch,
        )

    def forward(
        self,
        x,
        skip,
    ):

        x = F.interpolate(

            x,

            scale_factor=2,

            mode="bilinear",

            align_corners=False,
        )

        if x.shape[-2:] != skip.shape[-2:]:

            x = F.interpolate(

                x,

                size=skip.shape[-2:],

                mode="bilinear",

                align_corners=False,
            )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        x = self.conv1(x)

        x = self.conv2(x)

        return x


class ChangeFormer(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = timm.create_model(

            "swin_tiny_patch4_window7_224",

            pretrained=True,

            features_only=True,
        )

        encoder_channels = [
            96,
            192,
            384,
            768,
        ]

        # Fusion
        self.fuse1 = FusionBlock(
            96 * 2,
            96,
        )

        self.fuse2 = FusionBlock(
            192 * 2,
            192,
        )

        self.fuse3 = FusionBlock(
            384 * 2,
            384,
        )

        self.fuse4 = FusionBlock(
            768 * 2,
            768,
        )

        # Decoder
        self.dec3 = DecoderBlock(
            768,
            384,
            384,
        )

        self.dec2 = DecoderBlock(
            384,
            192,
            192,
        )

        self.dec1 = DecoderBlock(
            192,
            96,
            96,
        )

        self.head = nn.Sequential(

            ConvBNReLU(
                96,
                64,
            ),

            nn.Conv2d(
                64,
                1,
                kernel_size=1,
            ),
        )

    def reshape_feat(self, x):

        # Swin outputs BHWC
        if len(x.shape) == 4:

            x = x.permute(
                0,
                3,
                1,
                2,
            ).contiguous()

        return x

    def forward(
        self,
        pre,
        post,
    ):

        pre_feats = self.encoder(pre)

        post_feats = self.encoder(post.repeat(1,3,1,1))

        pre_feats = [
            self.reshape_feat(f)
            for f in pre_feats
        ]

        post_feats = [
            self.reshape_feat(f)
            for f in post_feats
        ]

        f1 = self.fuse1(
            pre_feats[0],
            post_feats[0],
        )

        f2 = self.fuse2(
            pre_feats[1],
            post_feats[1],
        )

        f3 = self.fuse3(
            pre_feats[2],
            post_feats[2],
        )

        f4 = self.fuse4(
            pre_feats[3],
            post_feats[3],
        )

        x = self.dec3(
            f4,
            f3,
        )

        x = self.dec2(
            x,
            f2,
        )

        x = self.dec1(
            x,
            f1,
        )

        x = F.interpolate(

            x,

            scale_factor=4,

            mode="bilinear",

            align_corners=False,
        )

        x = self.head(x)


        return x
