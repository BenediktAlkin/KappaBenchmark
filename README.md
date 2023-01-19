# KappaBenchmark

[![publish](https://github.com/BenediktAlkin/KappaBenchmark/actions/workflows/publish.yaml/badge.svg)](https://github.com/BenediktAlkin/KappaBenchmark/actions/workflows/publish.yaml)

Utilities for benchmarking [pytorch](https://pytorch.org/) applications.

- [Dataloading](https://github.com/BenediktAlkin/KappaBenchmark#dataloading)

## Setup
`pip install kappabenchmark`

## Dataloading

```
import kappabenchmark as kbm
dataloader = ...
result = kbm.benchmark_dataloading(
    dataloader=dataloader,
    num_epochs=...,
)
```

#### predefined benchmarks examples
- `python main_benchmark_grid.py --benchmark imagefolder --root ROOT --num_epochs 5 --batch_size 256 --num_workers 8,16 --num_fetch_workers 0,2,4`

#### register your own benchmark
write a script `run_mybenchmark.py`
```
import torch
from torch.utils.data import TensorDataset
from kappabenchmark.dataloading_benchmarks import DATALOADING_BENCHMARKS, DataloadingBenchmark
from kappabenchmark.scripts.main_benchmark_grid import parse_args, main

def mybenchmark(root):
    # root is a (optional) path to a directory which is passed via --root
    # for this toy dataset it is not needed
    return DataloadingBenchmark(dataset=TensorDataset(torch.randn(1024)))


if __name__ == "__main__":
    DATALOADING_BENCHMARKS["mybenchmark"] = mybenchmark
    main(**parse_args())
```
`python run_mybenchmark.py --benchmark mybenchmark [--root ROOT] --num_epochs 5 --batch_size 256 --num_workers 8,16 --num_fetch_workers 0,2,4`
