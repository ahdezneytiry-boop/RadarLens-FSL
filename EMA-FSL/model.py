import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarEncoder(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()
        self.fc = nn.Linear(input_dim, 64)

    def forward(self, x):
        x = x.mean(dim=1)
        return self.fc(x)


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)


class EMA_FSL(nn.Module):
    def __init__(self, num_classes=2, tau=0.7):
        super().__init__()
        self.tau = tau

        self.radar_encoder = RadarEncoder()
        self.visual_encoder = VisualEncoder()

        self.classifier_radar = nn.Linear(64, num_classes)
        self.classifier_fusion = nn.Linear(64 + 16, num_classes)

    def forward(self, radar, visual=None):
        radar_feat = self.radar_encoder(radar)

        radar_logits = self.classifier_radar(radar_feat)
        radar_prob = torch.softmax(radar_logits, dim=-1)
        confidence = radar_prob.max(dim=-1)[0]

        if visual is None or confidence.mean() >= self.tau:
            return radar_logits

        visual_feat = self.visual_encoder(visual)
        fusion = torch.cat([radar_feat, visual_feat], dim=1)
        return self.classifier_fusion(fusion)
