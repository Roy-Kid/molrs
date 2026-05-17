"""Domain validation checks for dielectric spectra.

All computation is in Rust; these are thin Python re-exports.
"""

from .molrs import (
    validate_conductivity_sum_rule_check as conductivity_sum_rule_check,
    validate_kramers_kronig_check as kramers_kronig_check,
    validate_route_agreement_check as route_agreement_check,
)
