# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from .base_strategy import BaseStrategy
from .mono_seq import MonoSeq
from .mono_rand import MonoRand
from .poly_seq import PolySeq
from .poly_int import PolyInt
from .poly_rand import PolyRand


STRATEGIES = {
    'mono_seq': MonoSeq,
    'mono_rand': MonoRand,
    'poly_seq': PolySeq,
    'poly_int': PolyInt,
    'poly_rand': PolyRand,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """Get a strategy instance by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)
