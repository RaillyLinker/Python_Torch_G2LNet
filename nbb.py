import torch
import torch.nn as nn
import torch.nn.functional as F


class G2LNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 글로벌 정보 -> 세부 정보
        self.comp_feat_blocks = nn.ModuleList([
            self._double_conv_block(1, 12, 8, 256, 256, 0),  # 256x256 -> 1x1
            self._double_conv_block(1 + 8, 24, 16, 128, 128, 0),  # 256x256 -> 2x2
            self._double_conv_block(1 + 16, 36, 24, 64, 64, 0),  # 256x256 -> 4x4
            self._double_conv_block(1 + 24, 48, 32, 32, 32, 0),  # 256x256 -> 8x8
            self._double_conv_block(1 + 32, 64, 48, 16, 16, 0),  # 256x256 -> 16x16
            self._double_conv_block(1 + 48, 80, 64, 8, 8, 0),  # 256x256 -> 32x32
            self._double_conv_block(1 + 64, 96, 80, 4, 4, 0),  # 256x256 -> 64x64
            self._double_conv_block(1 + 80, 112, 96, 3, 2, 1),  # 256x256 -> 128x128
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

    def forward(self, x):
        assert x.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

        result_feats_list = []
        x_h, x_w = x.shape[2], x.shape[3]

        # 컬러 특징 저장(128x128)
        result_feats_list.append(F.interpolate(x, scale_factor=0.5, mode='nearest'))

        # 분석을 위한 흑백 변환
        # 컬러 이미지는 그 자체로 색이란 특징을 지닌 특징 맵이고, 형태 특징을 구하기 위한 입력 값은 흑백으로 충분
        # 1x1 conv 를 굳이 할 필요 없이 검증된 알고리즘을 사용
        gray_feats = (0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :])

        # 멀티 스케일 형태 정보 추출
        x = gray_feats
        for i, block in enumerate(self.comp_feat_blocks):
            if i > 0:
                x = torch.cat([gray_feats, F.interpolate(x, size=(x_h, x_w), mode='nearest')], dim=1)
            x = block(x)

        # 추출된 특징 저장
        result_feats_list.append(x)

        # 특징 정보들 torch concat
        essential_feats = torch.cat(result_feats_list, dim=1)

        return essential_feats


# ----------------------------------------------------------------------------------------------------------------------
class UpsampleConcatClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = G2LNet()

        dummy_input = torch.zeros(2, 3, 256, 256)
        with torch.no_grad():
            backbone_output = self.backbone(dummy_input)

        _, backbone_output_ch, backbone_output_h, backbone_output_w = backbone_output.shape

        self.classifier = nn.Sequential(
            nn.Conv2d(backbone_output_ch, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Dropout(p=0.3),  # Dropout 추가
            nn.AdaptiveAvgPool2d(1),  # (B, 256, 1, 1)
            nn.Flatten(),  # (B, 256)
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        return x
