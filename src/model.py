from typing import Optional

import torch
import torchvision


class Retinanet_resnet50(torch.nn.Module):
    def __init__(self, num_classes: int, trainable_backbone_layers: int) -> None:
        super().__init__()
        self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            num_classes=num_classes,
            trainable_backbone_layers=trainable_backbone_layers,
            weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
        )

    def forward(
        self, images: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(images, targets)


if __name__ == "__main__":
    model = Retinanet_resnet50(2, trainable_backbone_layers=0)
    model.eval()
    output = model([torch.rand(1, 3, 512, 512)])

    print(output.shape)
