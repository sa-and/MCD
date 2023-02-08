"""This file describes all functions which can be used as cause-effect functions in an SCM"""

from typing import List
import random


# Some pre-defined functions. Additional functions for a specific use case can be added
def f_linear(parents: List[str]):
    weights = {p: random.uniform(-1, 1) for p in parents}

    def f(**kwargs):
        mu = 0.0
        for p in parents:
            mu += weights[p] * kwargs[p]
        return mu + random.gauss(0.0, random.uniform(0, 0.5))

    return f
