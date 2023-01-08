from itertools import product
from dataclasses import dataclass

@dataclass
class RunBenchmarkGridResult:
    parameters: dict
    variant_results: list

@dataclass
class VariantResult:
    name: str
    parameters: dict
    result: object

def grid_to_variants(**kwargs) -> list:
    # wrap parameters that are not lists/tuples
    for key in kwargs.keys():
        value = kwargs[key]
        if not isinstance(value, (list, tuple)):
            kwargs[key] = [value]
    # create variants
    variants = []
    for variant in product(*kwargs.values()):
        variants.append(dict(zip(kwargs.keys(), variant)))
    return variants

def variant_to_name(variant: dict) -> str:
    return " ".join(f"{key}={value}" for key, value in variant.items())

def run_benchmark_grid(param_grid: dict, run_fn, setup_fn=None, **kwargs) -> RunBenchmarkGridResult:
    variants = grid_to_variants(**param_grid)
    results = []
    for variant in variants:
        # noinspection PyBroadException
        try:
            if setup_fn is None:
                run_kwargs = {**variant, **kwargs}
            else:
                run_kwargs = setup_fn(**variant, **kwargs)
            result = run_fn(**run_kwargs)
        except Exception as e:
            result = str(e)
        results.append(VariantResult(
            name=variant_to_name(variant),
            parameters=variant,
            result=result,
        ))
    return RunBenchmarkGridResult(
        parameters=kwargs,
        variant_results=results,
    )