from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from time import sleep

import yaml
from pytorch_concurrent_dataloader import DataLoader as ConcurrentDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

from kappabenchmark.dataloading_benchmarks import DATALOADING_BENCHMARKS
from kappabenchmark.dataloading import benchmark_dataloading
from kappabenchmark.run import run_benchmark_grid


def parse_args():
    parser = ArgumentParser()
    # benchmark parameters
    parser.add_argument("--benchmark", type=str, choices=DATALOADING_BENCHMARKS.keys(), default="imagefolder")
    parser.add_argument("--root", type=str)
    # grid parameters
    duration_group = parser.add_mutually_exclusive_group(required=True)
    duration_group.add_argument("--num_epochs", type=str)
    duration_group.add_argument("--num_batches", type=str)
    parser.add_argument("--batch_size", type=str, required=True)
    parser.add_argument("--num_workers", type=str, required=True)
    parser.add_argument("--num_fetch_workers", type=str, default="0")
    parser.add_argument("--fetch_impl", type=str, choices=["asyncio", "threaded"], default="asyncio")
    # delay
    parser.add_argument("--initial_delay", type=int)
    return vars(parser.parse_args())


def setup_fn(dataset, collator, batch_size, num_workers, num_fetch_workers, fetch_impl, **kwargs):
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collator,
    )
    if num_fetch_workers != 0:
        dataloader = ConcurrentDataLoader(
            num_fetch_workers=num_fetch_workers,
            fetch_impl=fetch_impl,
            **dataloader_kwargs,
        )
    else:
        dataloader = TorchDataLoader(
            persistent_workers=num_workers > 0,
            **dataloader_kwargs,
        )

    return dict(dataloader=dataloader, **kwargs)


def parse_grid_param(param):
    if param is None:
        return None
    assert isinstance(param, str)
    return yaml.safe_load(f"[{param}]")


def on_variant_starts(i, count, name, **_):
    print(f"{i + 1}/{count}: {name}")


def on_variant_finished(variant_result, i, variant_count):
    print("----------------")
    print(f"{i + 1}/{variant_count}: {variant_result.name}")
    print("----------------")
    if isinstance(variant_result.result, str):
        print(f"FAILED: {variant_result.result}")
    else:
        for line in variant_result.result.to_string_lines():
            print(line)


def main(
        benchmark,
        root,
        num_epochs,
        num_batches,
        batch_size,
        num_workers,
        num_fetch_workers,
        fetch_impl,
        initial_delay,
):
    dataloading_benchmark = DATALOADING_BENCHMARKS[benchmark](root=str(Path(root).expanduser()))
    dataset = dataloading_benchmark.dataset
    collator = dataloading_benchmark.collator

    param_grid = {}
    if num_epochs is not None:
        param_grid["num_epochs"] = parse_grid_param(num_epochs)
    if num_batches is not None:
        param_grid["num_batches"] = parse_grid_param(num_batches)
    param_grid["batch_size"] = parse_grid_param(batch_size)
    param_grid["num_workers"] = parse_grid_param(num_workers)
    param_grid["num_fetch_workers"] = parse_grid_param(num_fetch_workers)
    param_grid["fetch_impl"] = parse_grid_param(fetch_impl)
    results = run_benchmark_grid(
        param_grid=param_grid,
        run_fn=partial(benchmark_dataloading, after_create_iter_fn=lambda: sleep(initial_delay or 0)),
        setup_fn=setup_fn,
        dataset=dataset,
        collator=collator,
        on_variant_starts=on_variant_starts,
        on_variant_finished=on_variant_finished,
    )
    print("----------------")
    print(f"total times")
    for i, result in enumerate(results.variant_results):
        if isinstance(result.result, str):
            result_str = result.result
        else:
            result_str = f"{result.result.num_samples} samples {result.result.total_time:.2f}"
        print(f"{i + 1}/{len(results.variant_results)} {result.name}: {result_str}")
