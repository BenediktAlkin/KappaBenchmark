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