# Physics — production-grade EGB perturbations

This document fully specifies the formulas implemented in
`src/deepegb/physics/egb_perturbations.py`. Every line below is tied to a
published reference; nothing is invented here. Follow the citations to verify
or to swap conventions.

## 1. Action and conventions

We work with the canonical four-derivative scalar–tensor action

$$
S = \int d^4x \sqrt{-g}\,\Bigl[\tfrac{M_\text{Pl}^2}{2}R
   - \tfrac{1}{2}(\partial\phi)^2
   - V(\phi)
   - \tfrac{1}{2}\xi(\phi)\,\mathcal G\Bigr],
\qquad
\mathcal G \equiv R^2 - 4 R_{\mu\nu}R^{\mu\nu} + R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}.
$$

We set $M_\text{Pl}=1$ throughout the code; restore factors of $M_\text{Pl}$
by $\xi\to\xi$, $\xi'\to\xi'/M_\text{Pl}$, and overall $1/M_\text{Pl}^2$ in
front of every $\xi$ contribution. Conformal time when needed is
$\tau$; cosmic time is $t$; primes on $V$ and $\xi$ denote $\partial/\partial\phi$.

This action and these conventions match
*Hwang & Noh 2005* ([gr-qc/0507025], henceforth **HN**),
*Koh, Lee, Tumurtushaa 2014* ([arXiv:1404.0027], **KLT**),
*Yi, Gong, Sabir 2018* ([arXiv:1811.01580], **YGS**),
and *Odintsov & Oikonomou 2018* ([arXiv:1810.04645], **OO**).

## 2. Background equations of motion

In a flat FLRW geometry with a homogeneous inflaton $\phi(t)$, the Einstein
and Klein–Gordon equations are (HN Eqs. 4–7, KLT Eqs. 5–8):

$$
3 H^2 = \tfrac{1}{2}\dot\phi^2 + V + 12 H^3 \dot\xi, \tag{F1}
$$

$$
2\dot H = -\dot\phi^2 + 4 H^2(\ddot\xi - H\dot\xi)
                       + 8 H \dot H\,\dot\xi, \tag{F2}
$$

$$
\ddot\phi + 3 H \dot\phi + V_{,\phi} + 12 H^2(\dot H + H^2)\xi_{,\phi} = 0. \tag{KG}
$$

In the slow-roll regime we drop $\ddot\phi$, $\ddot\xi$, and $\dot H$:

$$
3H^2 \approx V,\qquad
3H\dot\phi \approx -V_{,\phi} - 12 H^4 \xi_{,\phi}.
$$

Using $H^2 = V/3$ inside the second equation gives

$$
3H\dot\phi \approx -\bigl[V_{,\phi} + \tfrac{4}{3}V^2\xi_{,\phi}\bigr]
                  \equiv -Q(\phi),
\qquad
\boxed{\;Q(\phi) \equiv V_{,\phi} + \tfrac{4}{3} V^2\, \xi_{,\phi}\;}
\tag{Q}
$$

(matches **KLT Eq. 11** with their $\alpha = (4 V^2 \xi_{,\phi})/(3 V_{,\phi})$).

## 3. Slow-roll parameters

Following **KLT Eq. 9** and **YGS Eq. 2.10**:

$$
\varepsilon_1 \equiv -\dot H/H^2,\qquad
\varepsilon_2 \equiv \dot\varepsilon_1/(H\,\varepsilon_1),
$$

$$
\delta_1 \equiv 4 \dot\xi H/M_\text{Pl}^2,\qquad
\delta_2 \equiv \dot\delta_1/(H\,\delta_1).
$$

Substituting the slow-roll background:

$$
\varepsilon_1(\phi) = \frac{Q\,V_{,\phi}}{2\,V^2}, \qquad
\delta_1(\phi) = -\tfrac{4}{3}\,\xi_{,\phi}\,Q.
$$

(KLT Eq. 12, YGS Eq. 2.13.) The running parameters $\varepsilon_2$ and
$\delta_2$ are computed numerically along the slow-roll trajectory using
$dN/d\phi = -V/Q$ (KLT Eq. 13).

End of inflation is defined by $\varepsilon_1(\phi_\text{end}) = 1$. The
pivot field $\phi_N$ is found by integrating

$$
N(\phi) \;=\; \int_{\phi_\text{end}}^{\phi}\frac{V(\psi)}{Q(\psi)}\,d\psi.
$$

## 4. Tensor perturbations

Linearising $g_{ij} = a^2(\delta_{ij} + h_{ij}^\text{TT})$ in the EGB action
gives (HN Eq. 95, KLT Eq. 23, YGS Eq. 2.20):

$$
S_T = \tfrac{1}{8}\!\int dt\,d^3x\, a^3
\Bigl[ G_T (\dot h_{ij})^2
       - \tfrac{F_T}{a^2}(\partial_k h_{ij})^2 \Bigr],
$$

with

$$
G_T = M_\text{Pl}^2\bigl(1 - \delta_1\bigr),\qquad
F_T = M_\text{Pl}^2\bigl(1 - 4\ddot\xi/M_\text{Pl}^2\bigr).
$$

Using $\dot\xi = \delta_1\,M_\text{Pl}^2/(4H)$ and differentiating once
gives $4\ddot\xi/M_\text{Pl}^2 = \delta_1(\delta_2 + \varepsilon_1)$.
Hence the **tensor sound speed**

$$
\boxed{\;c_T^2(\phi) \;=\; \frac{F_T}{G_T}
   \;=\; \frac{1 - \delta_1(\delta_2 + \varepsilon_1)}{1 - \delta_1}\;.}
\tag{cT}
$$

The mode equation matched to Bunch–Davies gives the tensor power spectrum at
sound-horizon crossing $c_T k_\ast = a_\ast H_\ast$ (KLT Eq. 24, generalised):

$$
\boxed{\;P_T(k_\ast) \;=\; \frac{2\,H_\ast^2}{\pi^2 M_\text{Pl}^2}\,
   \frac{1}{(1-\delta_1)\,c_T^3}\Bigm|_{\phi=\phi_N}\;.}
\tag{PT}
$$

In the GR limit $\xi\to 0$ we have $\delta_1\to 0$, $c_T\to 1$, and (PT) reduces
to the textbook $P_T = 2H^2/(\pi^2 M_\text{Pl}^2)$.

## 5. Scalar perturbations

In the comoving gauge for the curvature perturbation $\mathcal R$, the
action is (HN Eq. 87, specialised):

$$
S_\mathcal R = \!\int dt\,d^3x\, a^3 Q_S\Bigl[
   \dot{\mathcal R}^2 - \tfrac{c_S^2}{a^2}(\partial\mathcal R)^2\Bigr].
$$

For the leading-order MVP we use the slow-roll-truncated amplitude
(KLT Eq. 19, YGS Eq. 3.12):

$$
\boxed{\;P_\mathcal R(k_\ast) \;=\; \frac{H_\ast^2}{8\pi^2 M_\text{Pl}^2\,\varepsilon_1}\Bigm|_{\phi=\phi_N}\;.}
\tag{PS}
$$

The scalar sound speed in single-field EGB inflation deviates from unity only
at $\mathcal{O}(\text{slow-roll}^2)$ (HN, App. C). We therefore set
$c_S^2 = 1$ in the MVP. The full HN expression (commented in
`compute_c_S2`) reads

$$
c_S^2 = 1 - \frac{8\,\dot\xi^2 H^2}{M_\text{Pl}^2 \varepsilon_1 \dot\phi^2 (1+\delta_1)^2}
              + \mathcal{O}(\text{slow-roll}^2),
$$

uncomment and use when targeting sub-percent precision on $n_s$.

## 6. Spectral indices and running

Rather than expanding $n_s, n_T$ as closed-form polynomials in
$\varepsilon_1, \delta_1, \delta_2, \eta, \dots$ — which differ between papers
at $\mathcal{O}(\text{slow-roll}^2)$ — we compute them as **numerical
N-derivatives** of $\ln P_\mathcal R$ and $\ln P_T$ along the inflationary
trajectory:

$$
n_s - 1 \;=\; \frac{d\ln P_\mathcal R}{d\ln k}\Bigm|_{k_\ast},
\qquad
n_T \;=\; \frac{d\ln P_T}{d\ln k}\Bigm|_{k_\ast}.
$$

Because $k_\ast = a_\ast H_\ast / c_T$ and $N$ counts e-folds *backwards
from end of inflation*, one has $d\ln k/dN_\text{pivot} = -1$ at leading
order, hence $d/d\ln k = -d/dN_\text{pivot}$. Concretely, for some small
$\Delta N$ (default $\Delta N = 0.5$):

$$
n_s - 1 \;=\; -\,\frac{\ln P_\mathcal R(\phi_{N+\Delta N})
                       - \ln P_\mathcal R(\phi_{N-\Delta N})}
                      {2\Delta N}.
$$

The running $\alpha_s = dn_s/d\ln k$ is the second numerical derivative
(the $-d/dN$ sign squares to $+d^2/dN^2$):

$$
\alpha_s \;=\; \frac{\ln P_\mathcal R(\phi_{N+\Delta N})
                     - 2\ln P_\mathcal R(\phi_N)
                     + \ln P_\mathcal R(\phi_{N-\Delta N})}
                    {(\Delta N)^2}.
$$

This captures all $\mathcal{O}(\text{slow-roll}^2)$ corrections automatically
provided the slow-roll background is accurate to that order — which it is
in our truncation.

## 7. Tensor-to-scalar ratio

$$
\boxed{\;r \;=\; \frac{P_T(k_\ast)}{P_\mathcal R(k_\ast)}\;.}
$$

In the GR limit this reduces to $r = 16\varepsilon_1$ as expected. In EGB
$r$ acquires a $1/[(1-\delta_1)\,c_T^3]$ enhancement and a numerator
modification through the GB-corrected $\varepsilon_1$. The **GR consistency
relation** $r = -8 n_T$ is broken by these factors — quantifying the breakage
is one way to confirm the EGB sector is on:

$$
\frac{r}{-8\,n_T} \;\ne\; 1 \quad\Longleftrightarrow\quad \xi \ne 0.
$$

This ratio is exposed as `consistency_r_minus_8nT` in `analyze_egb_model` and
plotted alongside the `(n_s, r)` star in `plot_egb_model`.

## 8. Validation against published results

Numerical sanity checks live in `tests/test_egb_perturbations.py`:

1. **GR Starobinsky** at $N=55$: produces $n_s = 0.96498$ vs.
   textbook $1 - 2/N = 0.96364$ (within $1.4\times10^{-3}$), $r=0.00350$
   vs. $12/N^2 = 0.00397$, and the consistency $r/(-8n_T) = 0.9999$.
2. **GR $m^2\phi^2/2$** at $N=60$: $n_s = 0.96694$ vs. $0.96667$
   (within $3\times10^{-4}$), $r = 0.13223$ vs. $8/N = 0.13333$,
   and the consistency $r/(-8n_T) = 1.0000$ to 4 digits.
3. **$c_T^2 = 1$ exactly** in the GR limit, computed directly by
   `compute_c_T2(model, phi)` at every $\phi$ on a sampled grid.
4. **Smooth EGB turn-on**: dialing $\xi\to 100\,\xi$ gives $\delta_1$ that
   scales linearly and a $c_T^2 - 1$ that scales linearly in $\delta_1$.

## 9. Full-kernel modules

The slow-roll closed-form kernel above is supplemented by three numerical
modules, each replacing one approximation:

### 9.1 `egb_background.py` — full background EOMs

Solves Eqs. (F1), (F2), (KG) numerically with `scipy.integrate.solve_ivp`
on $N = \ln(a/a_\text{init})$ as the integration variable, using state
$(\phi, \pi)$ with $\pi \equiv d\phi/dN$. The Friedmann constraint becomes
a *quadratic* in $H^2$ (since $\dot\phi = \pi H$):

$$
[12 \xi_{,\phi}\pi]\,H^4 + \bigl[\tfrac{\pi^2}{2} - 3\bigr]H^2 + V \;=\; 0,
$$

solved analytically and seeded with the GR root $H^2_\text{GR} = V/(3 - \pi^2/2)$.
At every step we solve a $2\times 2$ linear system for $(d\pi/dN,\ \varepsilon_1)$:

$$
\begin{pmatrix} 1 & -\pi - 12 H^2 \xi_{,\phi} \\
                4 H^2 \xi_{,\phi} & 2 - 12 H^2 \xi_{,\phi}\pi
\end{pmatrix}
\begin{pmatrix} d\pi/dN \\ \varepsilon_1 \end{pmatrix}
= \begin{pmatrix}
   -3\pi - V_{,\phi}/H^2 - 12 H^2 \xi_{,\phi} \\
   \pi^2(1 - 4 H^2 \xi_{,\phi\phi}) + 12 H^2 \xi_{,\phi}\pi
\end{pmatrix}.
$$

This recovers the slow-roll forms $\varepsilon_1 \to \pi^2/2$, $d\pi/dN + (3-\varepsilon_1)\pi + V_{,\phi}/H^2 \to 0$ in the GR limit.

End of inflation ($\varepsilon_1 = 1$) is detected by a `solve_ivp` event.
The output `BackgroundTrajectory` contains $N$, $\phi$, $\pi$, $H$, $\varepsilon_1$,
$\delta_1$, $a$, $\tau$ on a uniform $N$ grid.

### 9.2 `egb_modes.py` — Mukhanov–Sasaki integration

Integrates the canonical mode equations for $v_T = (a\sqrt{G_T}/\sqrt2)\,h$
and $v_S = z_S\,\mathcal R$ in the $N$ variable, using the smooth uniform
grid:

$$
\frac{d^2 v}{dN^2} + (1-\varepsilon_1)\frac{dv}{dN}
                  + \Bigl[\frac{c^2 k^2}{(aH)^2} - M(N)\Bigr] v = 0,
$$

with the effective mass

$$
M(N) = (1-\varepsilon_1)g_N + g_{NN} + g_N^2,
\qquad g \equiv \ln z = N + \tfrac{1}{2}\ln G_T \;\text{(or}\; \ln z_S^2\text{)}.
$$

For each $k$ we start at $N_\text{init}$ where $ck/(aH) \ge 50$ (deep
sub-Hubble), use Bunch–Davies initial conditions, and integrate to the end
of inflation. Power spectra are read off the late-time amplitude:

$$
P_T(k) = \frac{2 k^3}{\pi^2}\cdot \frac{4|v_T|^2}{a^2 G_T},
\qquad
P_\mathcal R(k) = \frac{k^3}{2\pi^2}\cdot \frac{|v_S|^2}{z_S^2}.
$$

**Validation**: At the pivot, MS-integrated $P_T$ matches the slow-roll
closed form to within 0.03%, $P_S$ to 1.5%, $n_s$ via finite differences
to 0.0001. The GR consistency $r = -8 n_T$ is reproduced to 10% from the
MS modes (limited by the $\Delta\ln k$ used in finite differencing).

### 9.3 `relic_gw.py` — relic GW spectrum

Given a primordial $P_T(k)$ from MS, computes today's relic GW energy
density:

$$
\Omega_\text{GW}(k)\,h^2 = \frac{1}{24}\,\Omega_R\,h^2\,P_T(k)\,\mathcal T(k)^2,
\qquad \Omega_R h^2 = 4.18\times 10^{-5}.
$$

The transfer function $\mathcal T^2(k)$ is

$$
\mathcal T^2(k) = \begin{cases}
   \tfrac{1}{2}\,\Bigl[g_*(T_\text{in})/g_{*,0}\Bigr]\,\Bigl[g_{*s,0}/g_{*s}(T_\text{in})\Bigr]^{4/3}
   & k > k_\text{eq} \quad\text{(RD modes)}\\[1mm]
   \tfrac{1}{2}\,(k_\text{eq}/k)^2 & k < k_\text{eq} \quad\text{(MD suppression)}
\end{cases}
$$

with $k_\text{eq} \approx 0.01$ Mpc⁻¹ and $g_*(T)$ interpolated between SM
thresholds (electron–positron annihilation, QCD, electroweak, top).

The inflation comoving wavenumber $k$ is mapped to today's $k$ in Mpc⁻¹ by
anchoring the pivot mode to the CMB pivot scale $k_* = 0.05$ Mpc⁻¹, then
to physical frequency by $f \approx 0.65\times 10^{-15}\,(k/\text{Mpc}^{-1})$ Hz.

References: Watanabe & Komatsu 2006 (astro-ph/0604176); Boyle & Steinhardt
2008 (astro-ph/0512014); Kuroyanagi et al. 2015 (1407.4785).

### 9.4 `egb_n3lo.py` — analytic N3LO observables (production analytic path)

The default path of `compute_observables_full` (since the N3LO upgrade) is
fully analytic with **exact** slow-roll coefficients — no WKB / uniform-
asymptotic residual:

1. **Exact sector reduction.** Both EGB perturbation sectors obey
   $\mu'' + (c^2 k^2 - z''/z)\mu = 0$ with exact $(z, c)$
   (Hwang–Noh 2005; Wu–Zhu–Wang, arXiv:1707.08020 Eqs. 2.8–2.12):
   scalar $z_R^2 = a^2(\dot\phi^2 + 6\bar\delta\dot\xi H^3)/[(1-\bar\delta/2)^2H^2]$,
   $\bar\delta = \delta_1/(1-\delta_1)$; tensor $z_h^2 = a^2(1-\delta_1)$,
   with the exact sound speeds $c_R^2$, $c_h^2$.  The sound-time
   transformation $d\varsigma = c\,d\eta$, $\tilde v = \sqrt{c}\,\mu$,
   $\tilde z = z\sqrt{c}$ maps each sector exactly onto the canonical
   problem $d^2\tilde v/d\varsigma^2 + (k^2 - \tilde z''/\tilde z)\tilde v = 0$
   (verified symbolically).

2. **Effective flow hierarchy from the full background.** With
   $\tilde{\mathcal H} = d\ln\tilde z/d\varsigma$:
   $\tilde\varepsilon_1 = 1 - \tilde{\mathcal H}'_\varsigma/\tilde{\mathcal H}^2$,
   $\tilde\varepsilon_{i+1} = d\ln\tilde\varepsilon_i/d\ln\tilde z$ —
   computed exactly (numerically) from the full Friedmann–KG trajectory.
   First-level derivatives are closed-form per point (no finite differences
   of solver output); higher flow functions come from a local polynomial
   fit of $\ln\tilde\varepsilon_1(\ln\tilde z)$.

3. **N3LO master formulas** (Auclair & Ringeval, arXiv:2205.12608,
   vendored as `_n3lo_master.py`): spectra exact through third order and
   indices through fourth order in the flow expansion, with the exact
   Green's-function constants $C = \gamma_E + \ln 2 - 2$, $\pi^2$,
   $\zeta(3)$.  Pivot at $-k\varsigma_\star = 1$ per sector;
   $P_S = \text{master}(\tilde H_S, \tilde\varepsilon_S)/8$,
   $P_T = \text{master}(\tilde H_T, \tilde\varepsilon_T)$ (normalisation
   fixed exactly by the GR limit).

Validation: `tests/test_egb_n3lo.py` pins the path against the
Mukhanov–Sasaki integrator (amplitudes ≲0.05%, tilts ≲1e-4 absolute) and
the GR textbook limits.  The corresponding closed-form expressions in
$(\varepsilon_i, \delta_i)$ — third-order tilts/runnings with exact
constants — are derived symbolically in
`scripts/derive_paper_formulas.py` and published in
[PAPER_FORMULAS.md](PAPER_FORMULAS.md) (+ `docs/paper_formulas.tex`,
machine-readable cache `outputs/paper_formulas_srepr.py`).  The third-order UAA formulas of Wu–Zhu–Wang are
implemented in `egb_uaa.py` as a literature cross-check only — the UAA
carries an irreducible ≈0.15% method residual ($181/36e^3$ normalisation,
$\ln 3$-type constants), which is why it is not used for inference.

**Corrections shipped with this upgrade** (affect all paths):

* The $\dot H$-equation coefficient in `egb_background._step_rhs` was
  wrong by a factor 3 on the $O(\delta_1)$ term
  ($+12 H^2\xi_{,\phi}\pi \to +4 H^2\xi_{,\phi}\pi$, i.e. the
  $-4\dot\xi H^3$ piece of F2).  The exact background identity
  $\dot\phi^2/H^2 = 2\varepsilon_1 - \delta_1 - \delta_1\varepsilon_1
  + \delta_1\delta_2$ now holds to machine precision (regression-tested).
* `compute_c_S2` now implements the exact scalar sound speed; the legacy
  approximation $c_S^2 = 1 - 4\xi_{,\phi}^2H^2/[\varepsilon_1(1-\delta_1)^2]$
  diverged for steep $\xi(\phi)$ (it could report $c_S^2 \approx -12$
  where the exact value is $1 - O(\delta_1^2)$).
* The Mukhanov–Sasaki scalar sector now uses the exact EGB $z_R$ for both
  the effective-mass term and the spectrum normalisation (previously the
  k-inflation form $2a^2\varepsilon_1/c_S^2$, an $O(\delta_1)$ bias).

## 9.5 GR-limit rejection (search-time)

The action reduces to plain GR when $\xi(\phi) \equiv 0$. Searching for EGB
inflation models in a space that includes the GR limit defeats the
purpose, so DeepEGB **rejects $\xi \to 0$ by default**.

The enforcement is a smooth penalty added to `chi2_full_breakdown`:

$$
\chi^2_{\rm EGB} = G \cdot \exp\bigl(-|\delta_1(\phi_N)|/\tau\bigr),
\qquad G = 10^3,\ \tau = 10^{-4}\ \text{(defaults)}.
$$

with $\delta_1 = 4 \dot\xi H/M_{\rm Pl}^2 |_{\phi_N}$. The penalty
vanishes for models with non-trivial GB sector at horizon crossing
$(|\delta_1| \gg \tau)$ and rises smoothly to $G$ as $|\delta_1| \to 0$,
giving PySR a usable gradient out of the GR basin. Configurable per
search via `SearchConfig.enforce_egb` (default `True`) and
`SearchConfig.egb_min_delta1` (default $10^{-4}$); also exposed on the
CLI as `--allow-gr` and `--egb-min-delta1`. The `diagnose_egb_model_tool`
flags any model with $|\delta_1| < 10^{-8}$ as `"is_gr_limit": true`.

In the symbolic-regression pipeline `run_joint_search` also drops the
`ξ = 0` candidate from the second-pass shortlist when `enforce_egb=True`,
so the GR baseline never enters the ranked output unless the user
explicitly opts in.

## 10. Remaining limitations

| Limitation | Magnitude | Where to fix |
|---|---|---|
| $c_S^2$ uses Kawai–Soda leading-order EGB form | $\mathcal O(\text{SR}^3)$ on $n_s$ | swap in full Horndeski formula via the now-available background trajectory. |
| Reheating modeled as instantaneous to RD | depends on duration of reheating | extend the trajectory through a parametric $w(N)$ epoch and re-integrate the modes. |
| Tensor mode integration assumes smooth post-inflation transfer | qualitative | full numerical integration of $h_k(\tau)$ from end of inflation through RD, MD, today. |
| No multi-field, hybrid exit, PBH-generating bumps | qualitative | new submodule. |

## 10. Citations

* HN — Hwang J., Noh H., *Cosmological perturbations in a generalized gravity
  including tachyonic condensation*, Phys.Rev.D **71** (2005) 063536,
  [gr-qc/0507025](https://arxiv.org/abs/gr-qc/0507025).
* KLT — Koh S., Lee B.-H., Tumurtushaa G., *Reconstruction of the scalar field
  potential in inflationary models with a Gauss–Bonnet term*, Phys.Rev.D
  **90** (2014) 063527, [arXiv:1404.0027](https://arxiv.org/abs/1404.0027).
* YGS — Yi Z., Gong Y., Sabir M., *Inflation with Gauss–Bonnet coupling*,
  Phys.Rev.D **98** (2018) 083521,
  [arXiv:1811.01580](https://arxiv.org/abs/1811.01580).
* OO — Odintsov S.D., Oikonomou V.K., *Viable inflation in scalar-Gauss–Bonnet
  gravity and reconstruction from observational indices*, Phys.Rev.D **98**
  (2018) 044039, [arXiv:1810.04645](https://arxiv.org/abs/1810.04645).
* Cartier C., Copeland E.J., Madden R., *The graceful exit in string
  cosmology*, JHEP **01** (2000) 035,
  [hep-th/9910169](https://arxiv.org/abs/hep-th/9910169) — earlier EGB
  inflation derivations.
