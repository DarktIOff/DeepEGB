# Vendored verbatim from the ancillary files of arXiv:2205.12608
# (Pierre Auclair & Christophe Ringeval, "Slow-Roll Inflation at N3LO",
# Phys. Rev. D 106, 063512). Original file: anc/minimal.py, v1.0 (2022/05).
# DeepEGB uses the *tensor-sector* functions as the generic N3LO master for
# the canonical mode equation d2v/ds2 + (k2 - z''/z) v = 0; see egb_n3lo.py.
# Do not edit: keep byte-identical to upstream below this header.
"""Slow-roll inflation at N3LO with minimal kinetic term.

Python module containing inflationary scalar and tensor slow-roll power spectra
at next-to-next-to-next to leading order, fully expanded around an observable pivot
wavenumber.
In this module, the pivot is chosen so that :math:`k_\\ast \\eta_\\ast = -1`.

Contains
========
- tensor_power_spectrum
- scalar_power_spectrum
- tensor_spectral_index
- scalar_spectral_index
- tensor_to_scalar_ratio

References
==========
All the formulas in this module are derived in the attached article [1].
At N2LO, they are consistent with the findings of [2].
.. [1] Pierre Auclair and Christophe Ringeval,
    ``Slow-Roll Inflation at N3LO''
.. [2] Jose Beltran Jimenez, Marcello Musso and Christophe Ringeval,
    ``Exact Mapping between Tensor and Most General Scalar Power Spectra'' (arXiv:1303.2788)
"""

__authors__ = ("Pierre Auclair", "Christophe Ringeval")
__contact__ = ("pierre.auclair@uclouvain.be", "christophe.ringeval@uclouvain.be")
__version__ = "1.0"
__date__ = "2022/05"

import numpy as np
import scipy.special as sf

CONSTANT_C = np.euler_gamma + np.log(2) - 2


def tensor_power_spectrum(
    wave_number, wave_number_ast, hubble_ast, eps_1_ast, eps_2_ast, eps_3_ast
):
    """Tensor power spectrum at N3LO.

    Parameters
    ----------
    - wave_number: float or numpy array
        Comoving wavenumber
    - wave_number_ast: float
        Pivot wavenumber
    - hubble_ast: float
        Hubble parameter at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_1_ast: float
        First Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_2_ast: float
        Second Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_3_ast: float
        Third Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`

    Returns
    -------
    Float or numpy array of dimensions size(wave_number)
    """
    return (
        2
        * hubble_ast ** 2
        / np.pi ** 2
        * (
            -(np.pi ** 2) * CONSTANT_C * eps_1_ast ** 3
            - 4 / 3 * CONSTANT_C ** 3 * eps_1_ast ** 3
            + 5 / 6 * np.pi ** 2 * CONSTANT_C * eps_1_ast ** 2 * eps_2_ast
            + 2 * CONSTANT_C ** 3 * eps_1_ast ** 2 * eps_2_ast
            + 1 / 12 * np.pi ** 2 * CONSTANT_C * eps_1_ast * eps_2_ast ** 2
            - 1 / 3 * CONSTANT_C ** 3 * eps_1_ast * eps_2_ast ** 2
            + 1 / 12 * np.pi ** 2 * CONSTANT_C * eps_1_ast * eps_2_ast * eps_3_ast
            - 1 / 3 * CONSTANT_C ** 3 * eps_1_ast * eps_2_ast * eps_3_ast
            + 13 / 12 * np.pi ** 2 * eps_1_ast ** 2 * eps_2_ast
            + 3 * CONSTANT_C ** 2 * eps_1_ast ** 2 * eps_2_ast
            + 1 / 12 * np.pi ** 2 * eps_1_ast * eps_2_ast ** 2
            - CONSTANT_C ** 2 * eps_1_ast * eps_2_ast ** 2
            + 1 / 12 * np.pi ** 2 * eps_1_ast * eps_2_ast * eps_3_ast
            - CONSTANT_C ** 2 * eps_1_ast * eps_2_ast * eps_3_ast
            + 1 / 2 * np.pi ** 2 * eps_1_ast ** 2
            + 2 * CONSTANT_C ** 2 * eps_1_ast ** 2
            + 8 * CONSTANT_C * eps_1_ast ** 3
            + 1 / 12 * np.pi ** 2 * eps_1_ast * eps_2_ast
            - CONSTANT_C ** 2 * eps_1_ast * eps_2_ast
            - 6 * CONSTANT_C * eps_1_ast ** 2 * eps_2_ast
            - 2 * CONSTANT_C * eps_1_ast * eps_2_ast ** 2
            - 2 * CONSTANT_C * eps_1_ast * eps_2_ast * eps_3_ast
            - 1
            / 3
            * (
                4 * eps_1_ast ** 3
                - 6 * eps_1_ast ** 2 * eps_2_ast
                + eps_1_ast * eps_2_ast ** 2
                + eps_1_ast * eps_2_ast * eps_3_ast
            )
            * np.log(wave_number / wave_number_ast) ** 3
            - 14 / 3 * eps_1_ast ** 3 * sf.zeta(3)
            - 2 / 3 * eps_1_ast * eps_2_ast ** 2 * sf.zeta(3)
            - 2 / 3 * eps_1_ast * eps_2_ast * eps_3_ast * sf.zeta(3)
            + 2 * CONSTANT_C * eps_1_ast ** 2
            + 16 / 3 * eps_1_ast ** 3
            - 2 * CONSTANT_C * eps_1_ast * eps_2_ast
            - 8 * eps_1_ast ** 2 * eps_2_ast
            - 2 / 3 * eps_1_ast * eps_2_ast ** 2
            - 2 / 3 * eps_1_ast * eps_2_ast * eps_3_ast
            - (
                4 * CONSTANT_C * eps_1_ast ** 3
                - 6 * CONSTANT_C * eps_1_ast ** 2 * eps_2_ast
                + CONSTANT_C * eps_1_ast * eps_2_ast ** 2
                + CONSTANT_C * eps_1_ast * eps_2_ast * eps_3_ast
                - 3 * eps_1_ast ** 2 * eps_2_ast
                + eps_1_ast * eps_2_ast ** 2
                + eps_1_ast * eps_2_ast * eps_3_ast
                - 2 * eps_1_ast ** 2
                + eps_1_ast * eps_2_ast
            )
            * np.log(wave_number / wave_number_ast) ** 2
            - 2 * CONSTANT_C * eps_1_ast
            - 3 * eps_1_ast ** 2
            - 2 * eps_1_ast * eps_2_ast
            - 1
            / 12
            * (
                12 * np.pi ** 2 * eps_1_ast ** 3
                + 48 * CONSTANT_C ** 2 * eps_1_ast ** 3
                - 10 * np.pi ** 2 * eps_1_ast ** 2 * eps_2_ast
                - 72 * CONSTANT_C ** 2 * eps_1_ast ** 2 * eps_2_ast
                - np.pi ** 2 * eps_1_ast * eps_2_ast ** 2
                + 12 * CONSTANT_C ** 2 * eps_1_ast * eps_2_ast ** 2
                - np.pi ** 2 * eps_1_ast * eps_2_ast * eps_3_ast
                + 12 * CONSTANT_C ** 2 * eps_1_ast * eps_2_ast * eps_3_ast
                - 72 * CONSTANT_C * eps_1_ast ** 2 * eps_2_ast
                + 24 * CONSTANT_C * eps_1_ast * eps_2_ast ** 2
                + 24 * CONSTANT_C * eps_1_ast * eps_2_ast * eps_3_ast
                - 48 * CONSTANT_C * eps_1_ast ** 2
                - 96 * eps_1_ast ** 3
                + 24 * CONSTANT_C * eps_1_ast * eps_2_ast
                + 72 * eps_1_ast ** 2 * eps_2_ast
                + 24 * eps_1_ast * eps_2_ast ** 2
                + 24 * eps_1_ast * eps_2_ast * eps_3_ast
                - 24 * eps_1_ast ** 2
                + 24 * eps_1_ast * eps_2_ast
                + 24 * eps_1_ast
            )
            * np.log(wave_number / wave_number_ast)
            - 2 * eps_1_ast
            + 1
        )
    )


def scalar_power_spectrum(
    wave_number, wave_number_ast, hubble_ast, eps_1_ast, eps_2_ast, eps_3_ast, eps_4_ast
):
    """Scalar power spectrum at N3LO.

    Parameters
    ----------
    - wave_number: float or numpy array
        Comoving wavenumber
    - wave_number_ast: float
        Pivot wavenumber
    - hubble_ast: float
        Hubble parameter at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_1_ast: float
        First Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_2_ast: float
        Second Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_3_ast: float
        Third Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_4_ast: float
        Fourth Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`

    Returns
    -------
    Float or numpy array of dimensions size(wave_number)
    """
    return (
        hubble_ast ** 2
        / (8 * np.pi ** 2 * eps_1_ast)
        * (
            -(np.pi ** 2) * CONSTANT_C * eps_1_ast ** 3
            - 4 / 3 * CONSTANT_C ** 3 * eps_1_ast ** 3
            - 2 / 3 * np.pi ** 2 * CONSTANT_C * eps_1_ast ** 2 * eps_2_ast
            - 1 / 4 * np.pi ** 2 * CONSTANT_C * eps_1_ast * eps_2_ast ** 2
            - 1 / 3 * CONSTANT_C ** 3 * eps_1_ast * eps_2_ast ** 2
            - 1 / 8 * np.pi ** 2 * CONSTANT_C * eps_2_ast ** 3
            - 1 / 6 * CONSTANT_C ** 3 * eps_2_ast ** 3
            + 1 / 2 * np.pi ** 2 * CONSTANT_C * eps_1_ast * eps_2_ast * eps_3_ast
            + 2 / 3 * CONSTANT_C ** 3 * eps_1_ast * eps_2_ast * eps_3_ast
            + 5 / 24 * np.pi ** 2 * CONSTANT_C * eps_2_ast ** 2 * eps_3_ast
            + 1 / 2 * CONSTANT_C ** 3 * eps_2_ast ** 2 * eps_3_ast
            + 1 / 24 * np.pi ** 2 * CONSTANT_C * eps_2_ast * eps_3_ast ** 2
            - 1 / 6 * CONSTANT_C ** 3 * eps_2_ast * eps_3_ast ** 2
            + 1 / 24 * np.pi ** 2 * CONSTANT_C * eps_2_ast * eps_3_ast * eps_4_ast
            - 1 / 6 * CONSTANT_C ** 3 * eps_2_ast * eps_3_ast * eps_4_ast
            + 13 / 12 * np.pi ** 2 * eps_1_ast ** 2 * eps_2_ast
            + 3 * CONSTANT_C ** 2 * eps_1_ast ** 2 * eps_2_ast
            + 5 / 8 * np.pi ** 2 * eps_1_ast * eps_2_ast ** 2
            + 1 / 2 * CONSTANT_C ** 2 * eps_1_ast * eps_2_ast ** 2
            + 1 / 12 * np.pi ** 2 * eps_1_ast * eps_2_ast * eps_3_ast
            - CONSTANT_C ** 2 * eps_1_ast * eps_2_ast * eps_3_ast
            + 1 / 2 * np.pi ** 2 * eps_1_ast ** 2
            + 2 * CONSTANT_C ** 2 * eps_1_ast ** 2
            + 8 * CONSTANT_C * eps_1_ast ** 3
            + 7 / 12 * np.pi ** 2 * eps_1_ast * eps_2_ast
            + CONSTANT_C ** 2 * eps_1_ast * eps_2_ast
            + 6 * CONSTANT_C * eps_1_ast ** 2 * eps_2_ast
            + 1 / 8 * np.pi ** 2 * eps_2_ast ** 2
            + 1 / 2 * CONSTANT_C ** 2 * eps_2_ast ** 2
            + CONSTANT_C * eps_1_ast * eps_2_ast ** 2
            + CONSTANT_C * eps_2_ast ** 3
            + 1 / 24 * np.pi ** 2 * eps_2_ast * eps_3_ast
            - 1 / 2 * CONSTANT_C ** 2 * eps_2_ast * eps_3_ast
            - 6 * CONSTANT_C * eps_1_ast * eps_2_ast * eps_3_ast
            - 2 * CONSTANT_C * eps_2_ast ** 2 * eps_3_ast
            - 1
            / 6
            * (
                8 * eps_1_ast ** 3
                + 2 * eps_1_ast * eps_2_ast ** 2
                + eps_2_ast ** 3
                - 4 * eps_1_ast * eps_2_ast * eps_3_ast
                - 3 * eps_2_ast ** 2 * eps_3_ast
                + eps_2_ast * eps_3_ast ** 2
                + eps_2_ast * eps_3_ast * eps_4_ast
            )
            * np.log(wave_number / wave_number_ast) ** 3
            - 14 / 3 * eps_1_ast ** 3 * sf.zeta(3)
            - 7 * eps_1_ast ** 2 * eps_2_ast * sf.zeta(3)
            - 25 / 6 * eps_1_ast * eps_2_ast ** 2 * sf.zeta(3)
            - 7 / 12 * eps_2_ast ** 3 * sf.zeta(3)
            - 2 / 3 * eps_1_ast * eps_2_ast * eps_3_ast * sf.zeta(3)
            - 1 / 3 * eps_2_ast * eps_3_ast ** 2 * sf.zeta(3)
            - 1 / 3 * eps_2_ast * eps_3_ast * eps_4_ast * sf.zeta(3)
            + 2 * CONSTANT_C * eps_1_ast ** 2
            + 16 / 3 * eps_1_ast ** 3
            - CONSTANT_C * eps_1_ast * eps_2_ast
            - 2 / 3 * eps_1_ast * eps_2_ast ** 2
            + 2 / 3 * eps_2_ast ** 3
            - 2 / 3 * eps_1_ast * eps_2_ast * eps_3_ast
            + 2 / 3 * eps_2_ast * eps_3_ast ** 2
            + 2 / 3 * eps_2_ast * eps_3_ast * eps_4_ast
            - 1
            / 2
            * (
                8 * CONSTANT_C * eps_1_ast ** 3
                + 2 * CONSTANT_C * eps_1_ast * eps_2_ast ** 2
                + CONSTANT_C * eps_2_ast ** 3
                - 4 * CONSTANT_C * eps_1_ast * eps_2_ast * eps_3_ast
                - 3 * CONSTANT_C * eps_2_ast ** 2 * eps_3_ast
                + CONSTANT_C * eps_2_ast * eps_3_ast ** 2
                + CONSTANT_C * eps_2_ast * eps_3_ast * eps_4_ast
                - 6 * eps_1_ast ** 2 * eps_2_ast
                - eps_1_ast * eps_2_ast ** 2
                + 2 * eps_1_ast * eps_2_ast * eps_3_ast
                - 4 * eps_1_ast ** 2
                - 2 * eps_1_ast * eps_2_ast
                - eps_2_ast ** 2
                + eps_2_ast * eps_3_ast
            )
            * np.log(wave_number / wave_number_ast) ** 2
            - 2 * CONSTANT_C * eps_1_ast
            - 3 * eps_1_ast ** 2
            - CONSTANT_C * eps_2_ast
            - 6 * eps_1_ast * eps_2_ast
            - eps_2_ast ** 2
            - 1
            / 24
            * (
                24 * np.pi ** 2 * eps_1_ast ** 3
                + 96 * CONSTANT_C ** 2 * eps_1_ast ** 3
                + 16 * np.pi ** 2 * eps_1_ast ** 2 * eps_2_ast
                + 6 * np.pi ** 2 * eps_1_ast * eps_2_ast ** 2
                + 24 * CONSTANT_C ** 2 * eps_1_ast * eps_2_ast ** 2
                + 3 * np.pi ** 2 * eps_2_ast ** 3
                + 12 * CONSTANT_C ** 2 * eps_2_ast ** 3
                - 12 * np.pi ** 2 * eps_1_ast * eps_2_ast * eps_3_ast
                - 48 * CONSTANT_C ** 2 * eps_1_ast * eps_2_ast * eps_3_ast
                - 5 * np.pi ** 2 * eps_2_ast ** 2 * eps_3_ast
                - 36 * CONSTANT_C ** 2 * eps_2_ast ** 2 * eps_3_ast
                - np.pi ** 2 * eps_2_ast * eps_3_ast ** 2
                + 12 * CONSTANT_C ** 2 * eps_2_ast * eps_3_ast ** 2
                - np.pi ** 2 * eps_2_ast * eps_3_ast * eps_4_ast
                + 12 * CONSTANT_C ** 2 * eps_2_ast * eps_3_ast * eps_4_ast
                - 144 * CONSTANT_C * eps_1_ast ** 2 * eps_2_ast
                - 24 * CONSTANT_C * eps_1_ast * eps_2_ast ** 2
                + 48 * CONSTANT_C * eps_1_ast * eps_2_ast * eps_3_ast
                - 96 * CONSTANT_C * eps_1_ast ** 2
                - 192 * eps_1_ast ** 3
                - 48 * CONSTANT_C * eps_1_ast * eps_2_ast
                - 144 * eps_1_ast ** 2 * eps_2_ast
                - 24 * CONSTANT_C * eps_2_ast ** 2
                - 24 * eps_1_ast * eps_2_ast ** 2
                - 24 * eps_2_ast ** 3
                + 24 * CONSTANT_C * eps_2_ast * eps_3_ast
                + 144 * eps_1_ast * eps_2_ast * eps_3_ast
                + 48 * eps_2_ast ** 2 * eps_3_ast
                - 48 * eps_1_ast ** 2
                + 24 * eps_1_ast * eps_2_ast
                + 48 * eps_1_ast
                + 24 * eps_2_ast
            )
            * np.log(wave_number / wave_number_ast)
            - 2 * eps_1_ast
            + 1
        )
    )


def tensor_spectral_index(
    eps_1_ast,
    eps_2_ast,
    eps_3_ast,
    eps_4_ast,
):
    """Scalar spectral index nt at N4LO.

    Parameters
    ----------
    - eps_1_ast: float
        First Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_2_ast: float
        Second Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_3_ast: float
        Third Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_4_ast: float
        Fourth Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`

    Returns
    -------
    Float or numpy array of dimensions size(wave_number)
    """
    return (
        2
        * (2 * np.pi ** 2 - 6 * CONSTANT_C - 7 * sf.zeta(3) - 14)
        * eps_1_ast ** 3
        * eps_2_ast
        + 1
        / 12
        * (
            31 * np.pi ** 2
            + 24 * (np.pi ** 2 - 16) * CONSTANT_C
            - 84 * CONSTANT_C ** 2
            - 360
        )
        * eps_1_ast ** 2
        * eps_2_ast ** 2
        - 1
        / 12
        * (
            4 * CONSTANT_C ** 3
            - np.pi ** 2
            - (np.pi ** 2 - 24) * CONSTANT_C
            + 12 * CONSTANT_C ** 2
            + 8 * sf.zeta(3)
            + 8
        )
        * eps_1_ast
        * eps_2_ast ** 3
        - 1
        / 12
        * (
            4 * CONSTANT_C ** 3
            - np.pi ** 2
            - (np.pi ** 2 - 24) * CONSTANT_C
            + 12 * CONSTANT_C ** 2
            + 8 * sf.zeta(3)
            + 8
        )
        * eps_1_ast
        * eps_2_ast
        * eps_3_ast ** 2
        - 1
        / 12
        * (
            4 * CONSTANT_C ** 3
            - np.pi ** 2
            - (np.pi ** 2 - 24) * CONSTANT_C
            + 12 * CONSTANT_C ** 2
            + 8 * sf.zeta(3)
            + 8
        )
        * eps_1_ast
        * eps_2_ast
        * eps_3_ast
        * eps_4_ast
        - 2 * eps_1_ast ** 4
        + (np.pi ** 2 - 6 * CONSTANT_C - 14) * eps_1_ast ** 2 * eps_2_ast
        + 1
        / 12
        * (np.pi ** 2 - 12 * CONSTANT_C ** 2 - 24 * CONSTANT_C - 24)
        * eps_1_ast
        * eps_2_ast ** 2
        + 1
        / 12
        * (np.pi ** 2 - 12 * CONSTANT_C ** 2 - 24 * CONSTANT_C - 24)
        * eps_1_ast
        * eps_2_ast
        * eps_3_ast
        - 2 * eps_1_ast ** 3
        - 2 * (CONSTANT_C + 1) * eps_1_ast * eps_2_ast
        - 2 * eps_1_ast ** 2
        + 1
        / 12
        * (
            4
            * (
                4 * np.pi ** 2
                + 3 * (np.pi ** 2 - 16) * CONSTANT_C
                - 12 * CONSTANT_C ** 2
                - 48
            )
            * eps_1_ast ** 2
            * eps_2_ast
            - 3
            * (
                4 * CONSTANT_C ** 3
                - np.pi ** 2
                - (np.pi ** 2 - 24) * CONSTANT_C
                + 12 * CONSTANT_C ** 2
                + 8 * sf.zeta(3)
                + 8
            )
            * eps_1_ast
            * eps_2_ast ** 2
        )
        * eps_3_ast
        - 2 * eps_1_ast
    )


def scalar_spectral_index(
    eps_1_ast,
    eps_2_ast,
    eps_3_ast,
    eps_4_ast,
    eps_5_ast,
):
    """Scalar spectral index ns - 1 at N4LO.

    Parameters
    ----------
    - eps_1_ast: float
        First Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_2_ast: float
        Second Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_3_ast: float
        Third Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_4_ast: float
        Fourth Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_5_ast: float
        Fifth Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`

    Returns
    -------
    Float or numpy array of dimensions size(wave_number)
    """
    return (
        (4 * np.pi ** 2 - 12 * CONSTANT_C - 14 * sf.zeta(3) - 29)
        * eps_1_ast ** 3
        * eps_2_ast
        + 1
        / 12
        * (
            61 * np.pi ** 2
            + 12 * (2 * np.pi ** 2 - 35) * CONSTANT_C
            - 84 * CONSTANT_C ** 2
            - 168 * sf.zeta(3)
            - 444
        )
        * eps_1_ast ** 2
        * eps_2_ast ** 2
        - 1
        / 24
        * (
            8 * CONSTANT_C ** 3
            - 21 * np.pi ** 2
            - 14 * (np.pi ** 2 - 12) * CONSTANT_C
            + 36 * CONSTANT_C ** 2
            + 100 * sf.zeta(3)
            + 88
        )
        * eps_1_ast
        * eps_2_ast ** 3
        + 1
        / 24
        * (np.pi ** 2 * CONSTANT_C - 4 * CONSTANT_C ** 3 - 8 * sf.zeta(3) + 16)
        * eps_2_ast
        * eps_3_ast ** 3
        + 1
        / 24
        * (np.pi ** 2 * CONSTANT_C - 4 * CONSTANT_C ** 3 - 8 * sf.zeta(3) + 16)
        * eps_2_ast
        * eps_3_ast
        * eps_4_ast ** 2
        + 1
        / 24
        * (np.pi ** 2 * CONSTANT_C - 4 * CONSTANT_C ** 3 - 8 * sf.zeta(3) + 16)
        * eps_2_ast
        * eps_3_ast
        * eps_4_ast
        * eps_5_ast
        - 2 * eps_1_ast ** 4
        + (np.pi ** 2 - 6 * CONSTANT_C - 15) * eps_1_ast ** 2 * eps_2_ast
        + 1
        / 12
        * (7 * np.pi ** 2 - 12 * CONSTANT_C ** 2 - 36 * CONSTANT_C - 84)
        * eps_1_ast
        * eps_2_ast ** 2
        + 1 / 24 * (np.pi ** 2 - 12 * CONSTANT_C ** 2) * eps_2_ast * eps_3_ast ** 2
        + 1
        / 24
        * (np.pi ** 2 - 12 * CONSTANT_C ** 2)
        * eps_2_ast
        * eps_3_ast
        * eps_4_ast
        - 2 * eps_1_ast ** 3
        - (2 * CONSTANT_C + 3) * eps_1_ast * eps_2_ast
        - CONSTANT_C * eps_2_ast * eps_3_ast
        + 1
        / 24
        * (
            12 * (np.pi ** 2 - 8) * CONSTANT_C * eps_2_ast ** 2
            - (
                8 * CONSTANT_C ** 3
                - 5 * np.pi ** 2
                - 2 * (7 * np.pi ** 2 - 72) * CONSTANT_C
                + 60 * CONSTANT_C ** 2
                + 16 * sf.zeta(3)
                + 16
            )
            * eps_1_ast
            * eps_2_ast
        )
        * eps_3_ast ** 2
        - 2 * eps_1_ast ** 2
        + 1
        / 12
        * (
            2
            * (
                17 * np.pi ** 2
                + 6 * (np.pi ** 2 - 19) * CONSTANT_C
                - 24 * CONSTANT_C ** 2
                - 42 * sf.zeta(3)
                - 120
            )
            * eps_1_ast ** 2
            * eps_2_ast
            - (
                12 * CONSTANT_C ** 3
                - 26 * np.pi ** 2
                - 21 * (np.pi ** 2 - 12) * CONSTANT_C
                + 60 * CONSTANT_C ** 2
                + 108 * sf.zeta(3)
                + 108
            )
            * eps_1_ast
            * eps_2_ast ** 2
            - 3 * eps_2_ast ** 3 * (7 * sf.zeta(3) - 8)
        )
        * eps_3_ast
        + 1
        / 12
        * (
            (7 * np.pi ** 2 - 12 * CONSTANT_C ** 2 - 48 * CONSTANT_C - 72)
            * eps_1_ast
            * eps_2_ast
            + 3 * (np.pi ** 2 - 8) * eps_2_ast ** 2
        )
        * eps_3_ast
        + 1
        / 24
        * (
            3
            * (np.pi ** 2 * CONSTANT_C - 4 * CONSTANT_C ** 3 - 8 * sf.zeta(3) + 16)
            * eps_2_ast
            * eps_3_ast ** 2
            + (
                6 * (np.pi ** 2 - 8) * CONSTANT_C * eps_2_ast ** 2
                - (
                    8 * CONSTANT_C ** 3
                    - 5 * np.pi ** 2
                    - 2 * (7 * np.pi ** 2 - 72) * CONSTANT_C
                    + 60 * CONSTANT_C ** 2
                    + 16 * sf.zeta(3)
                    + 16
                )
                * eps_1_ast
                * eps_2_ast
            )
            * eps_3_ast
        )
        * eps_4_ast
        - 2 * eps_1_ast
        - eps_2_ast
    )


def tensor_to_scalar_ratio(eps_1_ast, eps_2_ast, eps_3_ast, eps_4_ast):
    """Tensor to scalar ratio at N3LO.

    Parameters
    ----------
    - eps_1_ast: float
        First Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_2_ast: float
        Second Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_3_ast: float
        Third Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`
    - eps_4_ast: float
        Fourth Hubble flow function evaluated at :math:`\\eta_\\ast = - 1 / k_ast`

    Returns
    -------
    Float or numpy array of dimensions size(wave_number)
    """
    return (
        1
        / 12
        * (
            3 * np.pi ** 4
            + 16 * CONSTANT_C ** 4
            + 24 * (np.pi ** 2 - 8) * CONSTANT_C ** 2
            - 24 * CONSTANT_C ** 3
            - 66 * np.pi ** 2
            - 2 * (9 * np.pi ** 2 - 112 * sf.zeta(3) + 50) * CONSTANT_C
            + 168 * sf.zeta(3)
            + 144
        )
        * eps_1_ast ** 3
        * eps_2_ast
        + 1
        / 48
        * (
            17 * np.pi ** 4
            - 48 * CONSTANT_C ** 4
            - 40 * (np.pi ** 2 - 15) * CONSTANT_C ** 2
            - 96 * CONSTANT_C ** 3
            - 374 * np.pi ** 2
            - 24 * (13 * np.pi ** 2 - 42 * sf.zeta(3) - 60) * CONSTANT_C
            + 336 * sf.zeta(3)
            + 1488
        )
        * eps_1_ast ** 2
        * eps_2_ast ** 2
        + 1
        / 96
        * (
            13 * np.pi ** 4
            + 16 * CONSTANT_C ** 4
            - 8 * (11 * np.pi ** 2 - 120) * CONSTANT_C ** 2
            + 128 * CONSTANT_C ** 3
            - 224 * np.pi ** 2
            - 8 * (23 * np.pi ** 2 - 106 * sf.zeta(3) - 64) * CONSTANT_C
            + 112 * sf.zeta(3)
            + 832
        )
        * eps_1_ast
        * eps_2_ast ** 3
        + 1
        / 192
        * (
            3 * np.pi ** 4
            + 16 * CONSTANT_C ** 4
            - 48 * np.pi ** 2
            + 32 * CONSTANT_C * (7 * sf.zeta(3) - 8)
            + 192
        )
        * eps_2_ast ** 4
        - (np.pi ** 2 - CONSTANT_C - 7 * sf.zeta(3)) * eps_1_ast ** 2 * eps_2_ast
        - 1
        / 24
        * (
            19 * np.pi ** 2
            + 24 * (np.pi ** 2 - 9) * CONSTANT_C
            - 36 * CONSTANT_C ** 2
            - 84 * sf.zeta(3)
            - 48
        )
        * eps_1_ast
        * eps_2_ast ** 2
        + 1
        / 24
        * (
            4 * CONSTANT_C ** 3
            - 3 * (np.pi ** 2 - 8) * CONSTANT_C
            + 14 * sf.zeta(3)
            - 16
        )
        * eps_2_ast ** 3
        - 1
        / 24
        * (np.pi ** 2 * CONSTANT_C - 4 * CONSTANT_C ** 3 - 8 * sf.zeta(3) + 16)
        * eps_2_ast
        * eps_3_ast ** 2
        - 1
        / 24
        * (np.pi ** 2 * CONSTANT_C - 4 * CONSTANT_C ** 3 - 8 * sf.zeta(3) + 16)
        * eps_2_ast
        * eps_3_ast
        * eps_4_ast
        - 1 / 2 * (np.pi ** 2 - 2 * CONSTANT_C - 8) * eps_1_ast * eps_2_ast
        - 1 / 8 * (np.pi ** 2 - 4 * CONSTANT_C ** 2 - 8) * eps_2_ast ** 2
        - 1 / 24 * (np.pi ** 2 - 12 * CONSTANT_C ** 2) * eps_2_ast * eps_3_ast
        - 1
        / 576
        * (
            48
            * (
                np.pi ** 2 * CONSTANT_C ** 2
                - 4 * CONSTANT_C ** 4
                - 4 * CONSTANT_C ** 3
                + (np.pi ** 2 - 8 * sf.zeta(3) + 16) * CONSTANT_C
                - 8 * sf.zeta(3)
                + 16
            )
            * eps_1_ast
            * eps_2_ast
            - (
                np.pi ** 4
                - 72 * np.pi ** 2 * CONSTANT_C ** 2
                + 336 * CONSTANT_C ** 4
                + 384 * CONSTANT_C * (sf.zeta(3) - 2)
            )
            * eps_2_ast ** 2
        )
        * eps_3_ast ** 2
        - 1
        / 12
        * (
            (
                np.pi ** 2 * CONSTANT_C ** 2
                - 4 * CONSTANT_C ** 4
                - 4 * CONSTANT_C ** 3
                + (np.pi ** 2 - 8 * sf.zeta(3) + 16) * CONSTANT_C
                - 8 * sf.zeta(3)
                + 16
            )
            * eps_1_ast
            * eps_2_ast
            + (
                np.pi ** 2 * CONSTANT_C ** 2
                - 4 * CONSTANT_C ** 4
                - 8 * CONSTANT_C * (sf.zeta(3) - 2)
            )
            * eps_2_ast ** 2
        )
        * eps_3_ast
        * eps_4_ast
        + CONSTANT_C * eps_2_ast
        + 1
        / 288
        * (
            6
            * (
                np.pi ** 4
                - 48 * CONSTANT_C ** 4
                - 8 * (7 * np.pi ** 2 - 69) * CONSTANT_C ** 2
                + 48 * CONSTANT_C ** 3
                - 14 * np.pi ** 2
                - 4 * (13 * np.pi ** 2 - 96) * CONSTANT_C
            )
            * eps_1_ast ** 2
            * eps_2_ast
            + (
                13 * np.pi ** 4
                - 48 * CONSTANT_C ** 4
                - 24 * (25 * np.pi ** 2 - 228) * CONSTANT_C ** 2
                + 1152 * CONSTANT_C ** 3
                - 120 * np.pi ** 2
                - 48 * (5 * np.pi ** 2 - 4 * sf.zeta(3) - 28) * CONSTANT_C
            )
            * eps_1_ast
            * eps_2_ast ** 2
            + 3
            * (np.pi ** 4 - 60 * (np.pi ** 2 - 8) * CONSTANT_C ** 2 - 8 * np.pi ** 2)
            * eps_2_ast ** 3
        )
        * eps_3_ast
        - 1
        / 24
        * (
            2
            * (np.pi ** 2 + 6 * (np.pi ** 2 - 8) * CONSTANT_C - 12 * CONSTANT_C ** 2)
            * eps_1_ast
            * eps_2_ast
            - (12 * CONSTANT_C ** 3 - (7 * np.pi ** 2 - 48) * CONSTANT_C)
            * eps_2_ast ** 2
        )
        * eps_3_ast
        + 1
    )
