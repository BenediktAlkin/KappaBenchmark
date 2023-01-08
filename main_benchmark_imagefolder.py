from argparse import ArgumentParser
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader as TorchDataLoader
from pytorch_concurrent_dataloader import DataLoader as ConcurrentDataLoader
from kappabenchmark.dataloading import benchmark_dataloading
from pathlib import Path
from torchvision.transforms import CenterCrop, ToTensor, Compose
from tqdm import tqdm
from functools import partial

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument("--num_epochs", type=int)
    duration_group.add_argument("--num_batches", type=int)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--num_fetch_workers", type=int, default=0)
    return vars(parser.parse_args())

def start_pbar(num_batches, container):
    container.append(tqdm(total=num_batches))

def update_pbar(_, update_counter, container):
    container[0].update(update_counter)

def main(root, num_epochs, num_batches, batch_size, num_workers, num_fetch_workers):
    dataset = ImageFolder(
        root=str(Path(root).expanduser()),
        transform=Compose([
            CenterCrop(size=224),
            ToTensor(),
        ])
    )
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if num_fetch_workers > 0:
        dataloader = ConcurrentDataLoader(num_fetch_workers=num_fetch_workers, **dataloader_kwargs)
    else:
        dataloader = TorchDataLoader(**dataloader_kwargs)

    pbar_container = []
    result = benchmark_dataloading(
        dataloader=dataloader,
        num_epochs=num_epochs,
        num_batches=num_batches,
        total_num_batches_callback=partial(start_pbar, container=pbar_container),
        after_load_batch_callback=partial(update_pbar, container=pbar_container),
    )

    for line in result.to_string_lines():
        print(line)




if __name__ == "__main__":
    main(**parse_args())