from __future__ import annotations

import unittest

from src.retrieval.utils.args import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    parse_args,
)


class RetrievalArgsTestCase(unittest.TestCase):
    def test_parse_args_exposes_training_tuning_flags(self) -> None:
        args = parse_args(
            [
                "--batch-size",
                "256",
                "--weight-decay",
                "0.001",
                "--patience",
                "7",
                "--min-delta",
                "0.005",
            ]
        )

        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.weight_decay, 0.001)
        self.assertEqual(args.patience, 7)
        self.assertEqual(args.min_delta, 0.005)

    def test_parse_args_uses_expected_defaults(self) -> None:
        args = parse_args([])

        self.assertEqual(args.batch_size, DEFAULT_BATCH_SIZE)
        self.assertEqual(args.weight_decay, DEFAULT_WEIGHT_DECAY)
        self.assertEqual(args.patience, DEFAULT_PATIENCE)
        self.assertEqual(args.min_delta, DEFAULT_MIN_DELTA)


if __name__ == "__main__":
    unittest.main()
