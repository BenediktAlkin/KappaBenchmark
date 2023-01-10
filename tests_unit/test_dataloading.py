import unittest
from functools import partial

import torch
from torch.utils.data import DataLoader, TensorDataset

from kappabenchmark.dataloading import benchmark_dataloading


class TestDataloading(unittest.TestCase):
    def test_asserts(self):
        msg = "define benchmark duration via num_epochs or num_batches"
        with self.assertRaises(AssertionError) as ex:
            benchmark_dataloading(dataloader=None, num_epochs=1, num_batches=2)
        self.assertEqual(msg, str(ex.exception))
        with self.assertRaises(AssertionError) as ex:
            benchmark_dataloading(dataloader=None)
        self.assertEqual(msg, str(ex.exception))

    @staticmethod
    def increase_counter(counter):
        counter[0] += 1

    def test(self):
        dataset = TensorDataset(torch.randn(32))
        dataloader = DataLoader(dataset=dataset, batch_size=4, num_workers=0)
        create_iter_counter = [0]
        after_load_batch_counter = [0]
        after_create_iter_fn = partial(self.increase_counter, create_iter_counter)
        after_load_batch_fn = partial(self.increase_counter, after_load_batch_counter)
        result = benchmark_dataloading(
            dataloader=dataloader,
            num_epochs=1,
            after_create_iter_fn=after_create_iter_fn,
            after_load_batch_fn=after_load_batch_fn,
        )
        self.assertEqual(1, create_iter_counter[0])
        self.assertEqual(8, after_load_batch_counter[0])
        self.assertEqual(1, len(result.iter_times))
        self.assertEqual(8, len(result.batch_times))
