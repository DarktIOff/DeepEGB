"""
Catalogue of gravitational-wave detectors and their stochastic-background
sensitivities for use with the relic-GW spectrum.

Each entry stores

    * frequency band (Hz),
    * a representative peak frequency,
    * an order-of-magnitude Ω_GW h² floor — the SGWB amplitude that is
      detectable at SNR ~ 1 with a few-year mission. These are
      approximate; for forecasts use the published curves directly.
    * the era (current / planned / proposed),
    * the dominant probe (PTA / space / ground / CMB).

The numbers are drawn from the standard SGWB-forecast literature, in
particular:

* Caprini & Figueroa, *Cosmological backgrounds of gravitational waves*,
  CQG 35 (2018) 163001, arXiv:1801.04268
* Schmitz, *New Sensitivity Curves for GW Experiments*, JHEP 01 (2021) 097,
  arXiv:2002.04615
* Renzini, Goncharov, Jenkins & Meyers, *Stochastic Gravitational-Wave
  Backgrounds: Current Detection Efforts and Future Prospects*,
  Galaxies 10 (2022) 34, arXiv:2202.00178
* PTA collaborations (NANOGrav 15-yr, EPTA, IPTA): 2306.16213, 2306.16214
* LISA L3 mission concept; Robson, Cornish & Liu 2018 (arXiv:1803.01944)
* DECIGO concept paper, Kawamura et al. 2020 (arXiv:2006.13545)
* Einstein Telescope Design Report, Punturo et al. 2010
* Cosmic Explorer Horizon Study, Reitze et al. 2019, 2109.09882
* LiteBIRD, Hazumi et al. 2019; CMB-S4, Abazajian et al. 2016

If you need precise sensitivity curves at a given frequency, swap the
`sensitivity_at` floor with one of the published spectral curves
(e.g. via the `gwsense`/`pygwb` packages).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Detector:
    """A GW experiment with an approximate Ω_GW h² floor in its band."""

    name: str
    probe: str                          # "PTA", "space", "ground", "CMB"
    era: str                            # "current", "planned", "proposed"
    f_min_Hz: float
    f_max_Hz: float
    f_peak_Hz: float                    # frequency of best sensitivity
    omega_gw_h2_floor: float            # Ω_GW h² SNR≈1, integrated band
    description: str = ""
    references: tuple[str, ...] = field(default_factory=tuple)

    def covers(self, f_Hz: float) -> bool:
        return self.f_min_Hz <= f_Hz <= self.f_max_Hz

    def sensitivity(self, f_Hz: float | np.ndarray) -> np.ndarray:
        """Approximate Ω_GW h² floor at frequency f.  We use a parabola
        in log-space centred on f_peak — adequate for visualising whether
        a model lies above the detector's noise envelope.
        """
        f = np.asarray(f_Hz, dtype=float)
        # log-space parabola centred at f_peak, rising by 4 decades
        # at the band edges.
        log_f = np.log10(np.clip(f, 1e-30, None))
        log_peak = np.log10(self.f_peak_Hz)
        log_min = np.log10(self.f_min_Hz)
        log_max = np.log10(self.f_max_Hz)
        # half-width chosen so the curve hits ~1e4 × floor at the edges
        width = max(log_peak - log_min, log_max - log_peak, 0.5)
        rise = ((log_f - log_peak) / width) ** 2
        # Cap the exponent so we don't overflow at f far outside the band.
        sens = self.omega_gw_h2_floor * 10.0 ** np.minimum(rise * 4.0, 30.0)
        # Outside the band, return +inf so the curve looks "off-screen"
        out_of_band = (f < self.f_min_Hz) | (f > self.f_max_Hz)
        return np.where(out_of_band, np.inf, sens)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------
DETECTORS: tuple[Detector, ...] = (
    # --- Pulsar Timing Arrays ---
    Detector(
        name="NANOGrav 15yr", probe="PTA", era="current",
        f_min_Hz=1e-9, f_max_Hz=1e-7, f_peak_Hz=2e-8,
        omega_gw_h2_floor=2e-9,
        description="Operating since 2004; 67 ms-pulsars, 15-yr data set "
                    "(2306.16213). Reported evidence for a Hellings-Downs "
                    "stochastic signal at ~3 sigma.",
        references=("arXiv:2306.16213",),
    ),
    Detector(
        name="EPTA DR2", probe="PTA", era="current",
        f_min_Hz=1e-9, f_max_Hz=1e-7, f_peak_Hz=3e-8,
        omega_gw_h2_floor=3e-9,
        description="European PTA, 25-pulsar 24.7-yr data release (2306.16214).",
        references=("arXiv:2306.16214",),
    ),
    Detector(
        name="IPTA DR3", probe="PTA", era="planned",
        f_min_Hz=1e-9, f_max_Hz=1e-7, f_peak_Hz=2e-8,
        omega_gw_h2_floor=1e-9,
        description="International PTA combination of NANOGrav, EPTA, PPTA, "
                    "InPTA, MPTA. Targeting full Hellings-Downs detection.",
    ),
    Detector(
        name="SKA-PTA", probe="PTA", era="proposed",
        f_min_Hz=1e-10, f_max_Hz=1e-7, f_peak_Hz=1e-8,
        omega_gw_h2_floor=1e-15,
        description="Square Kilometre Array PTA. Best sensitivity to nanoHz "
                    "GWs of any planned facility.",
        references=("Janssen+ 2015, arXiv:1501.00127",),
    ),

    # --- Space-based interferometers ---
    Detector(
        name="LISA", probe="space", era="planned",
        f_min_Hz=1e-5, f_max_Hz=1e-1, f_peak_Hz=3e-3,
        omega_gw_h2_floor=1e-13,
        description="Laser Interferometer Space Antenna; ESA L3 mission, "
                    "launch ~2035. 2.5 Mkm triangular constellation.",
        references=("arXiv:1702.00786", "Robson+ 2018 1803.01944"),
    ),
    Detector(
        name="TaiJi", probe="space", era="planned",
        f_min_Hz=1e-4, f_max_Hz=1e-1, f_peak_Hz=3e-3,
        omega_gw_h2_floor=1e-13,
        description="Chinese Academy of Sciences space mission; arm length "
                    "~3 Mkm. Sensitivity comparable to LISA, complementary "
                    "constellation.",
        references=("Hu & Wu 2017",),
    ),
    Detector(
        name="TianQin", probe="space", era="planned",
        f_min_Hz=1e-4, f_max_Hz=1e0,  f_peak_Hz=3e-2,
        omega_gw_h2_floor=1e-12,
        description="Geocentric mission, ~10⁵ km arms. Higher peak f than "
                    "LISA, complementary band.",
        references=("Luo+ 2016",),
    ),
    Detector(
        name="LISA+TaiJi network", probe="space", era="planned",
        f_min_Hz=1e-5, f_max_Hz=1e-1, f_peak_Hz=3e-3,
        omega_gw_h2_floor=1e-14,
        description="Cross-correlation of LISA and TaiJi reduces noise by "
                    "another order of magnitude on the SGWB.",
        references=("Wang+ 2021, arXiv:2102.01708",),
    ),
    Detector(
        name="DECIGO", probe="space", era="proposed",
        f_min_Hz=1e-2, f_max_Hz=1e1, f_peak_Hz=1e-1,
        omega_gw_h2_floor=1e-17,
        description="Decihertz Observatory, ISAS-led. Bridges the LISA-LIGO "
                    "gap.",
        references=("Kawamura+ 2020, arXiv:2006.13545",),
    ),
    Detector(
        name="BBO", probe="space", era="proposed",
        f_min_Hz=1e-3, f_max_Hz=1e1, f_peak_Hz=3e-1,
        omega_gw_h2_floor=1e-17,
        description="Big Bang Observer, NASA concept successor to LISA. "
                    "Designed specifically for the inflationary SGWB.",
        references=("Crowder & Cornish 2005, gr-qc/0506015",),
    ),

    # --- Ground-based interferometers ---
    Detector(
        name="LIGO O4", probe="ground", era="current",
        f_min_Hz=2e1, f_max_Hz=4e3, f_peak_Hz=2e2,
        omega_gw_h2_floor=6e-9,
        description="Advanced LIGO O4 design sensitivity (2023+).",
        references=("arXiv:1304.0670",),
    ),
    Detector(
        name="aLIGO+VIRGO+KAGRA design", probe="ground", era="planned",
        f_min_Hz=1e1, f_max_Hz=4e3, f_peak_Hz=1e2,
        omega_gw_h2_floor=2e-9,
        description="Full design sensitivity of the international ground "
                    "network.",
    ),
    Detector(
        name="Einstein Telescope (ET)", probe="ground", era="planned",
        f_min_Hz=1e0, f_max_Hz=1e4, f_peak_Hz=1e2,
        omega_gw_h2_floor=1e-12,
        description="3rd-generation underground triangular interferometer, "
                    "10-km arms. Targeted construction 2030s.",
        references=("Punturo+ 2010, Hild+ 2011, arXiv:1012.0908",),
    ),
    Detector(
        name="Cosmic Explorer (CE)", probe="ground", era="proposed",
        f_min_Hz=5e0, f_max_Hz=4e3, f_peak_Hz=5e1,
        omega_gw_h2_floor=1e-12,
        description="L-shaped, 40-km arms, US-led 3rd-generation concept.",
        references=("Reitze+ 2019, arXiv:2109.09882",),
    ),

    # --- CMB B-mode polarisation experiments ---
    Detector(
        name="BICEP/Keck (current)", probe="CMB", era="current",
        f_min_Hz=1e-18, f_max_Hz=1e-16, f_peak_Hz=1e-17,
        omega_gw_h2_floor=2e-15,
        description="Constrains r at ~0.036 (95% CL, BK18). Acts as a probe "
                    "of inflationary GWs at horizon-re-entry frequencies "
                    "today.",
        references=("BICEP/Keck 2018, arXiv:2110.00483",),
    ),
    Detector(
        name="LiteBIRD", probe="CMB", era="planned",
        f_min_Hz=1e-18, f_max_Hz=1e-16, f_peak_Hz=1e-17,
        omega_gw_h2_floor=2e-16,
        description="JAXA satellite mission, target σ(r) ≈ 1e-3.",
        references=("Hazumi+ 2019",),
    ),
    Detector(
        name="CMB-S4", probe="CMB", era="planned",
        f_min_Hz=1e-18, f_max_Hz=1e-16, f_peak_Hz=1e-17,
        omega_gw_h2_floor=1e-16,
        description="Stage-4 ground-based CMB experiment, σ(r) ≈ 1e-3.",
        references=("Abazajian+ 2016, arXiv:1610.02743",),
    ),
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------
def detector_by_name(name: str) -> Detector | None:
    name_lc = name.lower()
    for d in DETECTORS:
        if d.name.lower() == name_lc:
            return d
    return None


def detectors_in_band(f_lo_Hz: float, f_hi_Hz: float) -> list[Detector]:
    """Return all detectors whose band overlaps [f_lo, f_hi]."""
    return [d for d in DETECTORS
            if not (d.f_max_Hz < f_lo_Hz or d.f_min_Hz > f_hi_Hz)]


def sensitivity_at(f_Hz: float | np.ndarray,
                   probes: Iterable[str] | None = None,
                   era: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """Return Ω_GW h² sensitivity floors at f for each detector."""
    out: dict[str, np.ndarray] = {}
    for d in DETECTORS:
        if probes and d.probe not in probes:
            continue
        if era and d.era not in era:
            continue
        out[d.name] = d.sensitivity(f_Hz)
    return out
