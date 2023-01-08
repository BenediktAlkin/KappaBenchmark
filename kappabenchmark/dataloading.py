from kappaprofiler import Stopwatch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from time import sleep
import math
from tqdm import tqdm

@dataclass
class BenchmarkDataloaderResult:
    num_workers: int
    prefetch_factor: int
    num_epochs: int
    num_batches: int
    num_samples: int
    batches_per_epoch: int
    total_time: float
    iter_times: list
    batch_times: list

    @property
    def total_iter_time(self):
        return sum(self.iter_times)

    @property
    def mean_iter_time(self):
        return self.total_iter_time / len(self.iter_times)

    @property
    def total_batch_time(self):
        return sum(self.batch_times)

    @property
    def mean_batch_time(self):
        return self.total_batch_time / len(self.batch_times)

    @property
    def batch_times_cleaned(self):
        # exclude first batch time from each worker
        # - time of first batch of the first worker is dependent on the batch_size
        return self.batch_times[self.num_workers:]

    @property
    def total_batch_time_cleaned(self):
        return sum(self.batch_times_cleaned)

    @property
    def mean_batch_time_cleaned(self):
        return self.total_batch_time_cleaned / len(self.batch_times_cleaned)

    def to_string_lines(self):
        lines = []
        if self.num_epochs is not None:
            lines.append(f"loaded {self.num_epochs} epochs")
        lines.append(f"loaded {self.num_batches} batches")
        lines.append(f"loaded {self.num_samples} samples")
        time_lines = [
            ("{}s total_time", self.total_time),
            ("{}s total_batch_time", self.total_batch_time),
            (f"{{}}s mean_batch_time ({self.num_batches} batches)", self.mean_batch_time),
            (f"{{}}s total_batch_time_cleaned (num_workers={self.num_workers})", self.total_batch_time_cleaned),
            (f"{{}}s mean_batch_time_cleaned (num_workers={self.num_workers})", self.mean_batch_time_cleaned),
        ]
        max_digits = max(int(math.log10(tl[1])) for tl in time_lines)
        format_str = f"{{:{max_digits+4}.2f}}"
        for i in range(len(time_lines)):
            lines.append(time_lines[i][0].format(format_str.format(time_lines[i][1])))
        return lines


def benchmark_dataloading(
        dataloader: DataLoader,
        num_epochs: int = None,
        num_batches: int = None,
        after_create_iter_fn=None,
        after_load_batch_fn=None,
):
    assert (num_batches is None) ^ (num_epochs is None), "define benchmark duration via num_epochs or num_batches"
    if num_batches is None:
        num_batches = num_epochs * len(dataloader)
    
    epoch_counter = 0
    batch_counter = 0
    sample_counter = 0
    stopwatch = Stopwatch()
    iter_times = []
    batch_times = []

    with tqdm(total=num_batches) as pbar:
        with Stopwatch() as total_sw:
            terminate = False
            while not terminate:
                # iterator
                with stopwatch:
                    dataloader_iter = iter(dataloader)
                iter_times.append(stopwatch.last_lap_time)
                if after_create_iter_fn is not None:
                    after_create_iter_fn()

                while True:
                    if batch_counter >= num_batches:
                        terminate = True
                        break
                    # load batch
                    try:
                        with stopwatch:
                            batch = next(dataloader_iter)
                        batch_times.append(stopwatch.last_lap_time)
                        sample_counter += len(batch)
                    except StopIteration:
                        break
                    if after_load_batch_fn is not None:
                        after_load_batch_fn()
                    batch_counter += 1
                    pbar.update(1)
                epoch_counter += 1
                if num_epochs is not None and epoch_counter >= num_epochs:
                    break

    return BenchmarkDataloaderResult(
        num_workers=dataloader.num_workers,
        prefetch_factor=dataloader.prefetch_factor,
        num_epochs=num_epochs,
        num_batches=num_batches,
        num_samples=sample_counter,
        batches_per_epoch=len(dataloader),
        total_time=total_sw.elapsed_time,
        iter_times=iter_times,
        batch_times=batch_times,
    )