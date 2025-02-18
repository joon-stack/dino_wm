from torchvision import transforms
import torch

def default_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )

def imagenet_transform(img_size=224):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

# def normalize(x: torch.Tensor) -> torch.Tensor:
#     if x.max() > 1.0:
#         x = x.float() / 255.0
#     # print("INFO: x.min, x.max: ", x.min(), x.max())
#     if x.shape[-1] == 3:
#         x = x.permute(2, 0, 1)
#     # print("INFO: x.shape: ", x.shape)
#     normalize = transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     return normalize(x)