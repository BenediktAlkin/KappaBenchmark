import unittest
from kappabenchmark.run import run_benchmark_grid, grid_to_variants, variant_to_name

class TestRun(unittest.TestCase):
    def test_grid_to_variants_scalar(self):
        variants = grid_to_variants(batch_size=[4, 8], num_workers=2)
        self.assertEqual(2, len(variants))
        expected = [
            dict(batch_size=4, num_workers=2),
            dict(batch_size=8, num_workers=2),
        ]
        self.assertEqual(expected, variants)

    def test_grid_to_variants_lists(self):
        variants = grid_to_variants(batch_size=[4, 8, 16], num_workers=[2, 4])
        self.assertEqual(6, len(variants))
        expected = [
            dict(batch_size=4, num_workers=2),
            dict(batch_size=4, num_workers=4),
            dict(batch_size=8, num_workers=2),
            dict(batch_size=8, num_workers=4),
            dict(batch_size=16, num_workers=2),
            dict(batch_size=16, num_workers=4),
        ]
        self.assertEqual(expected, variants)

    def test_variant_to_name(self):
        self.assertEqual("batch_size=5", variant_to_name(dict(batch_size=5)))
        self.assertEqual("batch_size=5 num_workers=3", variant_to_name(dict(batch_size=5, num_workers=3)))