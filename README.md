# KappaBenchmark

[![publish](https://github.com/BenediktAlkin/KappaBenchmark/actions/workflows/publish.yaml/badge.svg)](https://github.com/BenediktAlkin/KappaBenchmark/actions/workflows/publish.yaml)

Utilities for benchmarking [pytorch](https://pytorch.org/) applications.
- [Dataloading](https://github.com/BenediktAlkin/KappaBenchmark#dataloading)

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