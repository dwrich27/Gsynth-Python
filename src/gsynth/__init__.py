"""
gsynth — Generalized Synthetic Control Method in pure NumPy.

A Python port of the R ``gsynth`` and ``fect`` packages by Yiqing Xu
(Stanford) and collaborators.  The methodology and algorithms are entirely
their work; this package exists to make the method accessible in Python-first
workflows.  Please cite the original papers in any published research:

  Xu, Y. (2017). Generalized Synthetic Control Method. Political Analysis.
  Liu, L., Wang, Y., & Xu, Y. (2024). A Practical Guide to Counterfactual
      Estimators for Causal Inference with TSCS Data. AJPS.

Quick start
-----------
>>> import pandas as pd
>>> from gsynth import gsynth, plot, effect
>>>
>>> result = gsynth(
...     formula="Y ~ D",
...     data=df,
...     index=["unit", "year"],
...     force="unit",
...     CV=True,
...     se=True,
...     nboots=200,
...     seed=42,
... )
>>> print(result.summary())
>>> plot(result, type="gap")
>>> plot(result, type="counterfactual")
>>> plot(result, type="factors")
>>> eff = effect(result, cumu=True)

References
----------
Xu, Y. (2017). Generalized Synthetic Control Method: Causal Inference with
Interactive Fixed Effects Models. *Political Analysis*, 25(1), 57–76.
"""
from __future__ import annotations

from ._core import gsynth
from ._effect import effect
from ._plots import plot
from ._result import GsynthResult

__all__ = [
    "gsynth",
    "GsynthResult",
    "plot",
    "effect",
]

__version__ = "0.2.0"
__author__  = "dwrich27"
