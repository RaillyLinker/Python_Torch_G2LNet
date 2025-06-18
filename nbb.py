import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D


# todo : residual 에 이전 결과가 아니라 흑백 이미지를 추가하기
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_mask


class G2LNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 글로벌 정보 -> 세부 정보
        self.comp_feat_blocks = nn.ModuleList([
            self._double_conv_block_with_drop(1, 512, 512, 243, 243, 0, 11, 0.1),  # 243x243 -> 1x1
            self._double_conv_block_with_drop(512, 256, 256, 81, 81, 0, 9, 0.1),  # 243x243 -> 3x3
            self._double_conv_block_with_drop(256, 224, 224, 27, 27, 0, 7, 0.1),  # 243x243 -> 9x9
            self._double_conv_block_with_drop(224, 160, 160, 9, 9, 0, 7, 0.1),  # 243x243 -> 27x27
            self._double_conv_block(160, 128, 128, 3, 3, 0),  # 243x243 -> 81x81
            self._double_conv_block(128, 96, 96, 3, 1, 1),  # 243x243 -> 243x243
        ])

        # Residual 연결을 위한 1x1 conv 계층
        self.residual_convs = nn.ModuleList()
        for block in self.comp_feat_blocks:
            conv_layers = [l for l in block.children() if isinstance(l, nn.Conv2d)]
            in_ch = conv_layers[0].in_channels
            out_ch = conv_layers[-1].out_channels
            self.residual_convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))

        # Post-BN + Activation 블록
        self.post_bn_activations = nn.ModuleList()
        for block in self.comp_feat_blocks:
            conv_layers = [l for l in block.children() if isinstance(l, nn.Conv2d)]
            out_ch = conv_layers[-1].out_channels
            self.post_bn_activations.append(nn.Sequential(
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            ))

        num_blocks = len(self.comp_feat_blocks)
        drop_probs = [float(i) / num_blocks * 0.2 for i in range(num_blocks)]
        self.drop_paths = nn.ModuleList([
            StochasticDepth(p) for p in drop_probs
        ])

    def _double_conv_block(self, in_ch, mid_ch, out_ch, ks, strd, pdd):
        return nn.Sequential(
            # 평면당 형태를 파악
            nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),

            # 채널간 패턴 분석
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def _double_conv_block_with_drop(self, in_ch, mid_ch, out_ch, ks, strd, pdd, dbs, dbp):
        return nn.Sequential(
            # 평면당 형태를 파악
            nn.Conv2d(in_ch, mid_ch, kernel_size=ks, stride=strd, padding=pdd, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),
            DropBlock2D(block_size=dbs, drop_prob=dbp),

            # 채널간 패턴 분석
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

        result_feats_list = []
        x_h, x_w = x.shape[2], x.shape[3]

        # 컬러 특징 저장(243x243)
        result_feats_list.append(x)

        # 분석을 위한 흑백 변환(243x243)
        # 컬러 이미지는 그 자체로 색이란 특징을 지닌 특징 맵이고, 형태 특징을 구하기 위한 입력 값은 흑백으로 충분
        # 1x1 conv 를 굳이 할 필요 없이 검증된 알고리즘을 사용
        gray_feats = (0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :])

        # 멀티 스케일 형태 정보 추출
        x_in = gray_feats
        for i, (block, res_conv, drop_path, post_bn_act) in enumerate(
                zip(self.comp_feat_blocks, self.residual_convs, self.drop_paths, self.post_bn_activations)
        ):
            x_out = block(x_in)
            x_out = drop_path(x_out)
            x_out = F.interpolate(x_out, size=(x_h, x_w), mode='nearest')
            res = res_conv(x_in)
            x_in = x_out + res
            x_in = post_bn_act(x_in)
            result_feats_list.append(x_in)

        # 특징 정보들 torch concat
        essential_feats = torch.cat(result_feats_list, dim=1)

        return essential_feats


# ----------------------------------------------------------------------------------------------------------------------
class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = G2LNet()

        dummy_input = torch.zeros(2, 3, 243, 243)
        with torch.no_grad():
            backbone_output = self.backbone(dummy_input)

        _, backbone_output_ch, backbone_output_h, backbone_output_w = backbone_output.shape

        hidden_dim = backbone_output_ch * 2

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(backbone_output_ch),
            nn.Dropout(0.3),

            nn.Linear(backbone_output_ch, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x
