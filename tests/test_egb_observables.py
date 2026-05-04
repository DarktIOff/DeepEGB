"""(removed) The legacy leading-order toy kernel tests.

The kernel they exercised (`analyze_model`, `chi2_loss`, `Observables`) has
been removed in favour of the production stack:

    * `compute_observables_full`  — slow-roll closed-form with c_T², c_S².
    * `tensor_power_spectrum`     — full Mukhanov-Sasaki integration.
    * `relic_gw_spectrum`         — Ω_GW(f) h² with detector overlay.

See `tests/test_egb_perturbations.py` and `tests/test_full_kernel.py`.
"""
