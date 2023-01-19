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