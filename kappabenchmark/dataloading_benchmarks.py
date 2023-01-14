from dataclasses import dataclass

import kappadata as kd
import kappadata.common as common
import kappadata.transforms as transforms
import kappadata.wrappers.sample_wrappers as sample_wrappers
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop


@dataclass
class DataloadingBenchmark:
    dataset: Dataset
    collator: callable = None


def imagefolder_noaug_benchmark(root) -> DataloadingBenchmark:
    return DataloadingBenchmark(
        dataset=ImageFolder(
            root=root,
            transform=transforms.KDComposeTransform([
                CenterCrop(size=224),
                transforms.KDImageNetNorm(),
            ])
        ),
    )


def imagefolder_rrc_benchmark(root):
    return DataloadingBenchmark(
        dataset=ImageFolder(
            root=root,
            transform=transforms.KDComposeTransform([
                transforms.KDRandomResizedCrop(size=224),
                transforms.KDRandomHorizontalFlip(),
                transforms.KDImageNetNorm(),
            ])
        ),
    )


def imagefolder_byol_benchmark(root):
    return DataloadingBenchmark(
        dataset=kd.ModeWrapper(
            dataset=common.BYOLMultiViewWrapper(common.datasets.KDImageFolder(root=root)),
            mode="x",
        ),
    )


def imagefolder_maefinetune_benchmark(root):
    return DataloadingBenchmark(
        dataset=kd.ModeWrapper(
            dataset=sample_wrappers.LabelSmoothingWrapper(
                smoothing=0.1,
                dataset=common.datasets.KDImageFolder(
                    root=root,
                    transform=common.transforms.MAEFinetuneTransform(),
                ),
            ),
            mode="x class",
        ),
        collator=common.collators.MAEFinetuneMixCollator(),
    )


DATALOADING_BENCHMARKS = {
    "imagefolder_noaug": imagefolder_noaug_benchmark,
    "imagefolder_rrc": imagefolder_rrc_benchmark,
    "imagefolder_byol": imagefolder_byol_benchmark,
    "imagefolder_maefinetune": imagefolder_maefinetune_benchmark,
}
