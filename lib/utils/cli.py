"""Shared argparse validators."""

import argparse


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for budgets and worker counts."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer greater than zero") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed
