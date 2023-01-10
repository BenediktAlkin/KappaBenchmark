from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop, ToTensor, Compose


def imagefolder_benchmark(root):
    return ImageFolder(
        root=root,
        transform=Compose([
            CenterCrop(size=224),
            ToTensor(),
        ])
    )


BENCHMARKS = {
    "imagefolder": imagefolder_benchmark,
}
