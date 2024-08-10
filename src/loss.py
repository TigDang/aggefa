import torch
import torchvision
from torchvision.ops import distance_box_iou_loss


class DistanceBoxIoU_loss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, target, preds):
        return distance_box_iou_loss(target, preds, reduction="mean")
