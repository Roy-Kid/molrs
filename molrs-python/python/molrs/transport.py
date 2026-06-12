"""Ion-transport trajectory-analysis kernels.

All computation is in Rust. The kernels are grouped under three thin namespace
classes (static methods over the compiled ``molrs.molrs`` PyO3 functions) so
callers reach them as ``molrs.transport.Onsager.correlation(...)`` rather than
through a flat list of free functions.

These are the molrs ports of the *tame* recipes
(<https://github.com/Roy-Kid/tame>): ``onsager`` (Onsager transport
coefficients), ``jacf`` (current-ACF Green–Kubo conductivity), and ``persist``
(pair-survival / residence-time correlations).
"""

from .molrs import (
    transport_green_kubo_conductivity,
    transport_onsager_correlation,
    transport_pair_survival_tcf,
)


class Onsager:
    """Onsager collective mean-displacement cross-correlation (static)."""

    correlation = staticmethod(transport_onsager_correlation)


class Jacf:
    """Green–Kubo ionic conductivity from the charge-current ACF (static)."""

    green_kubo_conductivity = staticmethod(transport_green_kubo_conductivity)


class Persist:
    """Pair-survival (persistence) time-correlation functions (static)."""

    pair_survival_tcf = staticmethod(transport_pair_survival_tcf)


__all__ = ["Onsager", "Jacf", "Persist"]
