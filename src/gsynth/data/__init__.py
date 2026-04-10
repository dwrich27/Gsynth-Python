"""
Bundled datasets for gsynth examples and testing.

load_turnout()
    US voter turnout panel (1920–2012).  Used in Xu (2017) as the
    main empirical example for the Generalized Synthetic Control method.
"""
from __future__ import annotations

from importlib.resources import files


def load_turnout() -> "pandas.DataFrame":
    """
    Load the US voter turnout panel dataset.

    This is the canonical example dataset from Xu (2017) and the R gsynth
    package.  It covers 47 US states from 1920 to 2012 (every 4 years,
    24 time periods).  Nine states adopted Election Day Registration (EDR)
    at various points, creating a staggered adoption design.

    Columns
    -------
    abb           : str   — two-letter state abbreviation (unit ID)
    year          : int   — election year (time ID), 1920–2012 every 4 years
    turnout       : float — voter turnout (%) — outcome Y
    policy_edr    : int   — 1 if Election Day Registration is in effect (treatment D)
    policy_mail_in: int   — 1 if no-excuse absentee / mail-in voting is in effect
    policy_motor  : int   — 1 if motor-voter registration is in effect

    Treated states and adoption years
    ----------------------------------
    ME, MN, WI → 1976   |   ID, NH, WY → 1996
    IA, MT     → 2008   |   CT         → 2012

    Returns
    -------
    pandas.DataFrame  (1128 rows × 6 columns)

    Example
    -------
    >>> from gsynth.data import load_turnout
    >>> from gsynth import gsynth
    >>> df = load_turnout()
    >>> result = gsynth(
    ...     "turnout ~ policy_edr + policy_mail_in + policy_motor",
    ...     data=df, index=["abb", "year"], force="two-way", CV=True,
    ... )
    >>> print(f"ATT: {result.att_avg:.2f}")   # ≈ 4.9 (percentage points)

    Reference
    ---------
    Xu, Y. (2017). Generalized Synthetic Control Method. Political Analysis.
    """
    import pandas as pd
    path = files("gsynth.data").joinpath("turnout.csv")
    return pd.read_csv(path)
