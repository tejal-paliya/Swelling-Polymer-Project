"""
Kinetic analysis module for polymer bed swelling experiments.

Fits three established models from the literature to height vs. time data:

    1. Schott Pseudo-Second-Order (PSO)  — primary model for biopolymers
       Schott, H. (1992). J. Pharm. Sci. 81(5), 467-470.
       DOI: 10.1002/jps.2600810516

    2. Korsmeyer-Peppas Power Law        — diffusion mechanism diagnosis
       Ritger, P.L. & Peppas, N.A. (1987). J. Control. Release 5, 23-36.
       DOI: 10.1016/0168-3659(87)90034-4

    3. First-Order (Fickian) baseline    — comparison / null model
       Expected to fit poorly for stress-relaxation-controlled swelling.

Also computes:

    4. Flory-Rehner crosslink density    — equilibrium network characterisation
       Flory, P.J. & Rehner, J. (1943). J. Chem. Phys. 11, 521.
       DOI: 10.1063/1.1723792

All kinetic models operate on the dimensionless swelling degree S(t):

    S(t) = (h(t) - h_0) / h_0

where h_0 is the first valid height measurement in mm. Because the vial
has a fixed circular cross-section, height increase is proportional to
volumetric swelling, so no mass measurements are required.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats, optimize
from typing import Tuple, Dict, Optional
from pathlib import Path

from src.logger import setup_logger
from src.exceptions import DataError

logger = setup_logger("kinetic_analysis")

# ── Minimum data requirements ─────────────────────────────────────────────────
MIN_POINTS_REQUIRED = 10   # Minimum calibrated frames to attempt any fit
MIN_POINTS_POWER_LAW = 5   # Minimum points in first-60% window for Peppas fit


# ═════════════════════════════════════════════════════════════════════════════
#  1. DATA PREPARATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_swelling_degree(
    height_mm: np.ndarray,
    time_s: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute the dimensionless swelling degree S(t) from height measurements.

    S(t) = (h(t) - h_0) / h_0

    Args:
        height_mm: Array of bed height measurements in mm (smoothed recommended)
        time_s:    Corresponding timestamps in seconds

    Returns:
        Tuple of:
            - time_s_clean: Timestamps (seconds)
            - S:            Swelling degree array (dimensionless)
            - h0:           Initial height h_0 (mm)
            - Se:           Estimated equilibrium swelling degree (dimensionless)

    Raises:
        DataError: If data is insufficient or h_0 is zero/negative
    """
    if len(height_mm) < MIN_POINTS_REQUIRED:
        raise DataError(
            f"Kinetic analysis requires at least {MIN_POINTS_REQUIRED} calibrated "
            f"frames; only {len(height_mm)} were provided."
        )

    h0 = float(height_mm[0])
    if h0 <= 0:
        raise DataError(
            f"Initial height h_0 = {h0:.3f} mm is not positive. "
            "Cannot compute swelling degree."
        )

    S = (height_mm - h0) / h0

    # Estimate equilibrium swelling from the plateau (last 10% of time window)
    plateau_n = max(3, len(S) // 10)
    Se = float(np.mean(S[-plateau_n:]))

    logger.info(f"  h_0      = {h0:.2f} mm")
    logger.info(f"  Se       = {Se:.4f}  (estimated from last {plateau_n} points)")
    logger.info(f"  Max S(t) = {S.max():.4f}")

    return time_s.copy(), S, h0, Se


# ═════════════════════════════════════════════════════════════════════════════
#  2. POLYMER-SOLVENT INTERACTION CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════

def classify_interaction(S: np.ndarray) -> str:
    """
    Classify the polymer-solvent interaction type from the shape of S(t).

    This is called at the start of run_kinetic_analysis() before any model
    fitting. The result is written to kinetic_parameters.csv so that batch
    comparison scripts can filter experiments by interaction type.

    Classification logic:
        no_swelling  — S never rises above 0.01 (bad solvent, or too crosslinked)
        dissolution  — S(t) ends negative (polymer dissolving into solvent)
        gelation     — non-monotonic in first half: rises then falls as network forms
        dispersion   — excessive scatter / sign changes (granules disaggregating)
        swelling     — clean monotonic rise to plateau (target behaviour)

    Args:
        S: Swelling degree array (dimensionless)

    Returns:
        One of: 'no_swelling', 'dissolution', 'gelation', 'dispersion', 'swelling'
    """
    if S.max() < 0.01:
        return "no_swelling"

    if S[-1] < -0.02:
        return "dissolution"

    dS = np.diff(S)
    first_half = dS[:len(dS) // 2]

    # Gelation: significant downward movement in the first half of the experiment
    if np.sum(first_half < -0.005) > len(first_half) * 0.2:
        return "gelation"

    # Dispersion: excessive sign reversals indicating granules scattering
    sign_changes = np.sum(np.diff(np.sign(dS)) != 0)
    if sign_changes > len(dS) * 0.4:
        return "dispersion"

    return "swelling"


# ═════════════════════════════════════════════════════════════════════════════
#  3. CONCENTRATION CONVERSION
# ═════════════════════════════════════════════════════════════════════════════

def wv_pct_to_volume_fraction(
    concentration_wv_pct: float,
    polymer_density_g_per_cm3: float
) -> float:
    """
    Convert w/v% concentration to polymer volume fraction at preparation (phi_0).

    w/v% is defined as grams of polymer per 100 mL of solution.

        phi_0 = (c_g_per_mL) / rho_polymer

    This is the x-axis variable used in the literature when plotting Se or ks
    against preparation concentration (e.g. Yousefi et al., Polymer, 2005).

    Args:
        concentration_wv_pct:      Polymer concentration in w/v%
                                   (g polymer per 100 mL solution)
        polymer_density_g_per_cm3: Density of the dry polymer (g/cm3)
                                   e.g. PVA: ~1.26, chitosan: ~1.35, starch: ~1.50

    Returns:
        phi_0: Polymer volume fraction at preparation (dimensionless, 0–1)
    """
    c_g_per_mL = concentration_wv_pct / 100.0
    return c_g_per_mL / polymer_density_g_per_cm3


# ═════════════════════════════════════════════════════════════════════════════
#  4. FLORY-REHNER CROSSLINK DENSITY
# ═════════════════════════════════════════════════════════════════════════════

def compute_flory_rehner(
    h0_mm: float,
    he_mm: float,
    chi: float,
    V1_m3_per_mol: float = 1.8e-5
) -> Optional[float]:
    """
    Compute the effective crosslink density n (mol/m3) from the equilibrium
    swelling height using the Flory-Rehner equation.

    Theory:
        The Flory-Rehner equation balances the free energy of mixing
        (Flory-Huggins, governed by chi) against the elastic retractive
        force of the stretched network (rubber elasticity, governed by n):

            -[ ln(1-phi2) + phi2 + chi*phi2^2 ] = V1 * n * (phi2^(1/3) - phi2/2)

        Rearranged for n (what we solve for):

            n = -[ ln(1-phi2) + phi2 + chi*phi2^2 ] / [ V1 * (phi2^(1/3) - phi2/2) ]

        phi2 is derived from the height ratio:
            Qv   = he / h0        (volumetric swelling ratio)
            phi2 = 1 / Qv = h0 / he

    Args:
        h0_mm:          Initial bed height in mm
        he_mm:          Equilibrium bed height in mm (plateau mean from PSO)
        chi:            Flory-Huggins polymer-solvent interaction parameter
                        (dimensionless). Look up per polymer-solvent pair.
                        PVA-water: ~0.49-0.52, chitosan-water: ~0.42-0.50
        V1_m3_per_mol:  Molar volume of the solvent in m3/mol.
                        Default: water (1.8e-5 m3/mol).
                        Ethanol: 5.84e-5 m3/mol.

    Returns:
        n: Effective crosslink density in mol/m3, or None if outside the
           valid range of the model (phi2 >= 1, or rhs_factor <= 0).

    Note:
        Flory-Rehner is most reliable for covalently crosslinked networks.
        For physically crosslinked or entangled networks the result is an
        'apparent' crosslink density. Not applicable to dissolving polymers.
    """
    if he_mm <= 0 or h0_mm <= 0:
        logger.warning("  Flory-Rehner: invalid height values (must be > 0)")
        return None

    Qv   = he_mm / h0_mm          # volumetric swelling ratio
    phi2 = 1.0 / Qv               # polymer volume fraction in swollen state

    if phi2 >= 1.0:
        logger.warning(
            f"  Flory-Rehner: phi2={phi2:.4f} >= 1 — "
            "swelling ratio Qv <= 1, no swelling detected."
        )
        return None

    lhs        = -(np.log(1.0 - phi2) + phi2 + chi * phi2 ** 2)
    rhs_factor = phi2 ** (1.0 / 3.0) - phi2 / 2.0

    if rhs_factor <= 0:
        logger.warning(
            f"  Flory-Rehner: rhs_factor={rhs_factor:.6f} <= 0 — "
            "outside model validity range (phi2 too high)."
        )
        return None

    n = lhs / (V1_m3_per_mol * rhs_factor)

    logger.info(
        f"  Flory-Rehner: Qv={Qv:.4f}, phi2={phi2:.4f}, "
        f"n={n:.4f} mol/m3  (chi={chi}, V1={V1_m3_per_mol:.2e})"
    )
    return n


# ═════════════════════════════════════════════════════════════════════════════
#  5. MODEL FITTING
# ═════════════════════════════════════════════════════════════════════════════

def fit_schott_pso(
    time_s: np.ndarray,
    S: np.ndarray
) -> Dict:
    """
    Fit Schott's Pseudo-Second-Order (PSO) kinetic model via linear regression.

    Linearised form:
        t / S(t)  =  1 / (ks * Se^2)  +  t / Se

    A plot of (t/S) vs t gives:
        slope     = 1 / Se
        intercept = 1 / (ks * Se^2)

    So:
        Se  = 1 / slope
        ks  = slope^2 / intercept

    The initial swelling rate (dS/dt at t=0) is:
        initial_rate = ks * Se^2

    Args:
        time_s: Timestamps in seconds (must be > 0)
        S:      Swelling degree array

    Returns:
        Dictionary with keys:
            Se, ks, initial_rate, r_squared, slope, intercept,
            t_fit, t_over_S, t_over_S_fit, success, error_msg
    """
    result = {
        "model": "Schott PSO",
        "Se": np.nan,
        "ks": np.nan,
        "initial_rate": np.nan,
        "r_squared": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "t_fit": np.array([]),
        "t_over_S": np.array([]),
        "t_over_S_fit": np.array([]),
        "success": False,
        "error_msg": ""
    }

    # Only use points where t > 0 and S > 0 (avoid division by zero)
    mask = (time_s > 0) & (S > 0)
    t_valid = time_s[mask]
    S_valid = S[mask]

    if len(t_valid) < 3:
        result["error_msg"] = "Not enough valid points (t>0 and S>0) for PSO fit."
        logger.warning(f"  PSO: {result['error_msg']}")
        return result

    t_over_S = t_valid / S_valid
    slope, intercept, r, p_value, std_err = stats.linregress(t_valid, t_over_S)

    if slope <= 0 or intercept <= 0:
        result["error_msg"] = (
            f"PSO linear regression yielded non-physical parameters: "
            f"slope={slope:.6f}, intercept={intercept:.6f}. "
            "Data may not follow PSO kinetics."
        )
        logger.warning(f"  PSO: {result['error_msg']}")
        return result

    Se           = 1.0 / slope
    ks           = (slope ** 2) / intercept
    initial_rate = ks * Se ** 2   # = 1 / intercept
    t_over_S_fit = slope * t_valid + intercept

    result.update({
        "Se": Se,
        "ks": ks,
        "initial_rate": initial_rate,
        "r_squared": r ** 2,
        "slope": slope,
        "intercept": intercept,
        "t_fit": t_valid,
        "t_over_S": t_over_S,
        "t_over_S_fit": t_over_S_fit,
        "success": True
    })

    logger.info(f"  PSO fit:  Se={Se:.4f}, ks={ks:.6f}, R²={r**2:.4f}")
    return result


def fit_korsmeyer_peppas(
    time_s: np.ndarray,
    S: np.ndarray,
    Se: float
) -> Dict:
    """
    Fit the Korsmeyer-Peppas power law model via log-log linear regression.

    Model (applied to first 60% of swelling only):
        S(t) / Se  =  k * t^n

    Log-linearised form:
        ln(S/Se) = ln(k) + n * ln(t)

    The diffusional exponent n identifies transport mechanism:
        n = 0.5   → Fickian diffusion
        0.5 < n < 1 → Anomalous (non-Fickian) transport
        n = 1     → Case II (relaxation-controlled)
        n > 1     → Super Case II

    Args:
        time_s: Timestamps in seconds
        S:      Swelling degree array
        Se:     Equilibrium swelling degree

    Returns:
        Dictionary with keys:
            n, k_pp, r_squared, mechanism, mask_60,
            log_t_fit, log_frac_fit, success, error_msg
    """
    result = {
        "model": "Korsmeyer-Peppas",
        "n": np.nan,
        "k_pp": np.nan,
        "r_squared": np.nan,
        "mechanism": "unknown",
        "mask_60": np.array([], dtype=bool),
        "log_t_fit": np.array([]),
        "log_frac_fit": np.array([]),
        "success": False,
        "error_msg": ""
    }

    frac = S / Se

    # Apply to first 60% of swelling only; exclude t=0 and S<=0
    mask = (frac > 0.01) & (frac <= 0.60) & (time_s > 0)
    result["mask_60"] = mask

    if mask.sum() < MIN_POINTS_POWER_LAW:
        result["error_msg"] = (
            f"Only {mask.sum()} points fall within the first 60% of swelling "
            f"(need >= {MIN_POINTS_POWER_LAW}). Try a longer experiment or more frames."
        )
        logger.warning(f"  Peppas: {result['error_msg']}")
        return result

    log_t    = np.log(time_s[mask])
    log_frac = np.log(frac[mask])
    slope, intercept, r, p_value, std_err = stats.linregress(log_t, log_frac)

    n    = slope
    k_pp = np.exp(intercept)

    if n < 0.45:
        mechanism = "Fickian diffusion (n < 0.45)"
    elif n <= 0.89:
        mechanism = "Anomalous / non-Fickian transport (0.45 <= n <= 0.89)"
    else:
        mechanism = "Case II or Super-Case II transport (n > 0.89)"

    log_t_fit    = np.linspace(log_t.min(), log_t.max(), 100)
    log_frac_fit = slope * log_t_fit + intercept

    result.update({
        "n": n,
        "k_pp": k_pp,
        "r_squared": r ** 2,
        "mechanism": mechanism,
        "log_t_fit": log_t_fit,
        "log_frac_fit": log_frac_fit,
        "success": True
    })

    logger.info(f"  Peppas:   n={n:.4f}, k={k_pp:.6f}, R²={r**2:.4f}  ->  {mechanism}")
    return result


def fit_first_order(
    time_s: np.ndarray,
    S: np.ndarray,
    Se: float
) -> Dict:
    """
    Fit the first-order (Fickian) kinetic model via non-linear least squares.

    Model:
        S(t) = Se * (1 - exp(-k1 * t))

    This is expected to fit WORSE than PSO for biopolymers. It is included
    as a comparison baseline — a lower R² here confirms the stress-relaxation
    mechanism identified by the PSO model.

    Args:
        time_s: Timestamps in seconds
        S:      Swelling degree array
        Se:     Equilibrium swelling degree (used as initial guess, not fixed)

    Returns:
        Dictionary with keys:
            k1, Se_fo, r_squared, t_dense, S_fit, success, error_msg
    """
    result = {
        "model": "First-Order",
        "k1": np.nan,
        "Se_fo": np.nan,
        "r_squared": np.nan,
        "t_dense": np.array([]),
        "S_fit": np.array([]),
        "success": False,
        "error_msg": ""
    }

    mask    = time_s >= 0
    t_valid = time_s[mask]
    S_valid = S[mask]

    def first_order_model(t, k1, Se_param):
        return Se_param * (1.0 - np.exp(-k1 * t))

    try:
        popt, _ = optimize.curve_fit(
            first_order_model,
            t_valid,
            S_valid,
            p0=[0.005, Se],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=10000
        )
        k1_fit, Se_fo = popt

        S_predicted = first_order_model(t_valid, k1_fit, Se_fo)
        ss_res    = np.sum((S_valid - S_predicted) ** 2)
        ss_tot    = np.sum((S_valid - np.mean(S_valid)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        t_dense = np.linspace(t_valid.min(), t_valid.max(), 300)
        S_fit   = first_order_model(t_dense, k1_fit, Se_fo)

        result.update({
            "k1": k1_fit,
            "Se_fo": Se_fo,
            "r_squared": r_squared,
            "t_dense": t_dense,
            "S_fit": S_fit,
            "success": True
        })
        logger.info(f"  1st-order: k1={k1_fit:.6f}, Se={Se_fo:.4f}, R²={r_squared:.4f}")

    except (RuntimeError, ValueError) as e:
        result["error_msg"] = f"First-order curve_fit failed: {e}"
        logger.warning(f"  1st-order: {result['error_msg']}")

    return result


# ═════════════════════════════════════════════════════════════════════════════
#  6. OUTPUTS — CSV AND MULTI-PANEL PLOT
# ═════════════════════════════════════════════════════════════════════════════

def save_kinetic_results(
    time_s: np.ndarray,
    S: np.ndarray,
    Se: float,
    h0: float,
    pso: Dict,
    peppas: Dict,
    first_order: Dict,
    output_dir: str,
    interaction_type: str = "swelling",
    concentration_wv_pct: Optional[float] = None,
    phi0: Optional[float] = None,
    n_crosslink: Optional[float] = None,
    chi: Optional[float] = None,
) -> Tuple[str, str]:
    """
    Save kinetic analysis results: parameters CSV and multi-panel figure.

    Args:
        time_s:               Timestamps in seconds
        S:                    Swelling degree array
        Se:                   Estimated equilibrium swelling degree
        h0:                   Initial height in mm
        pso:                  PSO fit result dict from fit_schott_pso()
        peppas:               Peppas fit result dict from fit_korsmeyer_peppas()
        first_order:          First-order fit result dict from fit_first_order()
        output_dir:           Directory to save files
        interaction_type:     Classification string from classify_interaction()
        concentration_wv_pct: Polymer concentration in w/v% (optional)
        phi0:                 Polymer volume fraction at preparation (optional)
        n_crosslink:          Flory-Rehner crosslink density in mol/m3 (optional)
        chi:                  Flory-Huggins parameter used for FR calculation (optional)

    Returns:
        Tuple of (csv_path, plot_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    def _val(x, decimals=4):
        """Return rounded value or 'N/A' for None, 'FAILED' for nan."""
        if x is None:
            return "N/A"
        if isinstance(x, float) and np.isnan(x):
            return "FAILED"
        return round(x, decimals)

    # ── Parameter CSV ─────────────────────────────────────────────────────────
    rows = [
        # ── Experiment metadata ───────────────────────────────────────────────
        {"Parameter": "interaction_type",
         "Value": interaction_type,
         "Model": "General",
         "Notes": "Polymer-solvent interaction classification"},
        {"Parameter": "h_0_mm",
         "Value": _val(h0),
         "Model": "General",
         "Notes": "Initial bed height (mm)"},
        {"Parameter": "Se_estimated",
         "Value": _val(Se),
         "Model": "General",
         "Notes": "Equilibrium swelling degree (plateau mean)"},
        {"Parameter": "n_data_points",
         "Value": len(S),
         "Model": "General",
         "Notes": "Total calibrated frames used"},
        {"Parameter": "concentration_wv_pct",
         "Value": _val(concentration_wv_pct, 2) if concentration_wv_pct is not None else "N/A",
         "Model": "General",
         "Notes": "Polymer concentration (g per 100 mL)"},
        {"Parameter": "phi_0_preparation",
         "Value": _val(phi0, 4) if phi0 is not None else "N/A",
         "Model": "General",
         "Notes": "Polymer volume fraction at preparation"},

        # ── Schott PSO ────────────────────────────────────────────────────────
        {"Parameter": "Se_pso",
         "Value": _val(pso["Se"]) if pso["success"] else "FAILED",
         "Model": "Schott PSO",
         "Notes": "Equilibrium swelling from PSO slope"},
        {"Parameter": "ks",
         "Value": _val(pso["ks"], 6) if pso["success"] else "FAILED",
         "Model": "Schott PSO",
         "Notes": "PSO rate constant (1/s)"},
        {"Parameter": "initial_rate_pso",
         "Value": _val(pso["initial_rate"], 6) if pso["success"] else "FAILED",
         "Model": "Schott PSO",
         "Notes": "Initial swelling rate = ks * Se^2 (1/s)"},
        {"Parameter": "R2_pso",
         "Value": _val(pso["r_squared"]) if pso["success"] else "FAILED",
         "Model": "Schott PSO",
         "Notes": "Coefficient of determination"},

        # ── Korsmeyer-Peppas ──────────────────────────────────────────────────
        {"Parameter": "n_peppas",
         "Value": _val(peppas["n"]) if peppas["success"] else "FAILED",
         "Model": "Korsmeyer-Peppas",
         "Notes": "Diffusional exponent (mechanism indicator)"},
        {"Parameter": "k_peppas",
         "Value": _val(peppas["k_pp"], 6) if peppas["success"] else "FAILED",
         "Model": "Korsmeyer-Peppas",
         "Notes": "Power law pre-factor"},
        {"Parameter": "mechanism",
         "Value": peppas["mechanism"] if peppas["success"] else "FAILED",
         "Model": "Korsmeyer-Peppas",
         "Notes": "Transport mechanism classification"},
        {"Parameter": "R2_peppas",
         "Value": _val(peppas["r_squared"]) if peppas["success"] else "FAILED",
         "Model": "Korsmeyer-Peppas",
         "Notes": "Coefficient of determination"},

        # ── First-Order ───────────────────────────────────────────────────────
        {"Parameter": "k1",
         "Value": _val(first_order["k1"], 6) if first_order["success"] else "FAILED",
         "Model": "First-Order",
         "Notes": "First-order rate constant (1/s)"},
        {"Parameter": "Se_first_order",
         "Value": _val(first_order["Se_fo"]) if first_order["success"] else "FAILED",
         "Model": "First-Order",
         "Notes": "Equilibrium swelling from first-order fit"},
        {"Parameter": "R2_first_order",
         "Value": _val(first_order["r_squared"]) if first_order["success"] else "FAILED",
         "Model": "First-Order",
         "Notes": "Coefficient of determination"},

        # ── Flory-Rehner ──────────────────────────────────────────────────────
        {"Parameter": "chi",
         "Value": _val(chi) if chi is not None else "N/A",
         "Model": "Flory-Rehner",
         "Notes": "Flory-Huggins interaction parameter (user-supplied)"},
        {"Parameter": "n_crosslink_mol_per_m3",
         "Value": _val(n_crosslink) if n_crosslink is not None else "N/A",
         "Model": "Flory-Rehner",
         "Notes": "Effective crosslink density (mol/m3)"},
    ]

    params_df = pd.DataFrame(rows)
    csv_path  = os.path.join(output_dir, "kinetic_parameters.csv")
    params_df.to_csv(csv_path, index=False)
    logger.info(f"  Saved kinetic parameters: {csv_path}")

    # ── Swelling degree time-series CSV ───────────────────────────────────────
    swelling_df = pd.DataFrame({
        "time_s":              time_s,
        "swelling_degree_S":   S,
        "fractional_S_over_Se": S / Se if Se != 0 else np.nan
    })
    swelling_csv = os.path.join(output_dir, "swelling_degree.csv")
    swelling_df.to_csv(swelling_csv, index=False)
    logger.info(f"  Saved swelling degree series: {swelling_csv}")

    # ── Multi-panel figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Polymer Swelling Kinetics — Model Fitting  "
        f"[{interaction_type.replace('_', ' ').title()}]",
        fontsize=16, fontweight="bold", y=0.98
    )
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, :])   # Top row: S(t) with model fits
    ax2 = fig.add_subplot(gs[1, 0])   # Bottom-left: PSO linearisation
    ax3 = fig.add_subplot(gs[1, 1])   # Bottom-right: Peppas log-log

    # Panel 1 — S(t) with model curves
    ax1.scatter(time_s, S, s=18, alpha=0.45, color="#4C72B0",
                label="S(t) measured", zorder=2)
    ax1.axhline(Se, color="grey", linestyle="--", linewidth=1.2,
                label=f"Se (estimated) = {Se:.3f}")

    if pso["success"]:
        t_dense = np.linspace(time_s[time_s > 0].min(), time_s.max(), 500)
        ks_val, Se_pso = pso["ks"], pso["Se"]
        S_pso = (ks_val * Se_pso ** 2 * t_dense) / (1.0 + ks_val * Se_pso * t_dense)
        ax1.plot(t_dense, S_pso, color="#D62728", linewidth=2.2,
                 label=f"PSO fit  (Se={Se_pso:.3f}, ks={ks_val:.4f}, R²={pso['r_squared']:.3f})")

    if first_order["success"]:
        ax1.plot(first_order["t_dense"], first_order["S_fit"],
                 color="#2CA02C", linewidth=1.8, linestyle="-.",
                 label=f"1st-order  (k1={first_order['k1']:.4f}, R²={first_order['r_squared']:.3f})")

    # Annotate concentration on the plot if available
    if concentration_wv_pct is not None:
        ax1.text(
            0.02, 0.97,
            f"Concentration: {concentration_wv_pct:.2f} w/v%"
            + (f"  (phi_0={phi0:.4f})" if phi0 is not None else ""),
            transform=ax1.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
        )

    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Swelling Degree S(t)", fontsize=12)
    ax1.set_title("Swelling Degree vs Time — PSO and First-Order Fits", fontsize=13)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.25)

    # Panel 2 — PSO linearisation t/S vs t
    if pso["success"]:
        ax2.scatter(pso["t_fit"], pso["t_over_S"], s=20, alpha=0.5,
                    color="#4C72B0", label="t/S data", zorder=2)
        ax2.plot(pso["t_fit"], pso["t_over_S_fit"], color="#D62728", linewidth=2,
                 label=f"Linear fit  R²={pso['r_squared']:.3f}\n"
                       f"Slope=1/Se  ->  Se={pso['Se']:.3f}")
        ax2.set_xlabel("Time t (s)", fontsize=12)
        ax2.set_ylabel("t / S(t)  (s)", fontsize=12)
        ax2.set_title(
            "PSO Linearisation: t/S vs t\n"
            r"(slope = 1/S$_e$,  intercept = 1/(k$_s \cdot$S$_e^2$))",
            fontsize=12
        )
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.25)
    else:
        ax2.text(0.5, 0.5, f"PSO fit failed:\n{pso['error_msg']}",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=10, color="red", wrap=True)
        ax2.set_title("PSO Linearisation (failed)", fontsize=12)

    # Panel 3 — Korsmeyer-Peppas log-log
    if peppas["success"]:
        mask_plot = peppas["mask_60"]
        ax3.scatter(np.log(time_s[mask_plot]), np.log(S[mask_plot] / Se),
                    s=20, alpha=0.55, color="#4C72B0",
                    label=f"Data (S/Se <= 0.60,  {mask_plot.sum()} pts)", zorder=2)
        ax3.plot(peppas["log_t_fit"], peppas["log_frac_fit"],
                 color="#FF7F0E", linewidth=2,
                 label=f"Power law fit\nn={peppas['n']:.3f},  R²={peppas['r_squared']:.3f}")

        x_ref = np.array([peppas["log_t_fit"].min(), peppas["log_t_fit"].max()])
        mid_y = np.mean(peppas["log_frac_fit"])
        ax3.plot(x_ref, 0.5 * (x_ref - x_ref.mean()) + mid_y,
                 color="green", linestyle=":", linewidth=1.2, alpha=0.7,
                 label="n=0.5 (Fickian)")
        ax3.plot(x_ref, 1.0 * (x_ref - x_ref.mean()) + mid_y,
                 color="purple", linestyle=":", linewidth=1.2, alpha=0.7,
                 label="n=1.0 (Case II)")

        ax3.set_xlabel("ln(t)", fontsize=12)
        ax3.set_ylabel("ln(S/Se)", fontsize=12)
        ax3.set_title(
            f"Korsmeyer-Peppas Log-Log Plot\n"
            f"Mechanism: {peppas['mechanism'].split('(')[0].strip()}",
            fontsize=12
        )
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.25)
    else:
        ax3.text(0.5, 0.5, f"Peppas fit failed:\n{peppas['error_msg']}",
                 ha="center", va="center", transform=ax3.transAxes,
                 fontsize=10, color="red", wrap=True)
        ax3.set_title("Korsmeyer-Peppas (failed)", fontsize=12)

    plot_path = os.path.join(output_dir, "kinetic_fit_plot.png")
    fig.savefig(plot_path, dpi=900, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved kinetic plot: {plot_path}")

    return csv_path, plot_path


# ═════════════════════════════════════════════════════════════════════════════
#  7. PUBLIC ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def run_kinetic_analysis(
    height_mm: np.ndarray,
    time_s: np.ndarray,
    output_dir: str,
    concentration_wv_pct: Optional[float] = None,
    polymer_density_g_per_cm3: Optional[float] = None,
    chi: Optional[float] = None,
    V1_m3_per_mol: float = 1.8e-5,
) -> Optional[Tuple[str, str]]:
    """
    Run the full kinetic analysis pipeline on a height vs. time dataset.

    Called from height_inference.py after measurements are collected, or
    directly from test_kinetic_analysis.py for offline testing.

    Args:
        height_mm:                 1D array of smoothed bed height measurements (mm).
        time_s:                    1D array of corresponding timestamps (seconds).
        output_dir:                Directory where output files will be written.
        concentration_wv_pct:      Polymer concentration in w/v% (optional).
                                   Pass params['concentration_wv_pct'] from main.py.
        polymer_density_g_per_cm3: Dry polymer density (g/cm3) for phi_0 conversion.
                                   Required if concentration_wv_pct is provided.
                                   PVA: ~1.26, chitosan: ~1.35, starch: ~1.50
        chi:                       Flory-Huggins interaction parameter for
                                   Flory-Rehner crosslink density calculation.
                                   If None, Flory-Rehner is skipped.
        V1_m3_per_mol:             Molar volume of solvent (m3/mol).
                                   Default: water (1.8e-5).

    Returns:
        Tuple of (csv_path, plot_path) if successful, or None on failure.

    Outputs written to output_dir:
        - kinetic_parameters.csv    All fitted parameters + FR + concentration
        - swelling_degree.csv       S(t) time-series
        - kinetic_fit_plot.png      Three-panel figure
    """
    logger.info("\n=== Starting Kinetic Analysis ===")
    logger.info(f"  Data points : {len(height_mm)}")
    logger.info(f"  Time range  : {time_s.min():.1f} - {time_s.max():.1f} s")
    logger.info(f"  Height range: {height_mm.min():.2f} - {height_mm.max():.2f} mm")
    if concentration_wv_pct is not None:
        logger.info(f"  Concentration: {concentration_wv_pct:.2f} w/v%")

    height_mm = np.asarray(height_mm, dtype=float)
    time_s    = np.asarray(time_s,    dtype=float)

    if len(height_mm) != len(time_s):
        logger.error("height_mm and time_s have different lengths — aborting.")
        return None

    # ── Step 1: Compute swelling degree ──────────────────────────────────────
    try:
        time_s_clean, S, h0, Se = compute_swelling_degree(height_mm, time_s)
    except DataError as e:
        logger.error(f"Kinetic analysis aborted: {e}")
        return None

    # ── Step 2: Classify interaction type ────────────────────────────────────
    interaction_type = classify_interaction(S)
    logger.info(f"\n  Interaction type: {interaction_type.upper()}")

    if interaction_type in ("no_swelling", "dissolution"):
        logger.warning(
            f"  Skipping kinetic model fitting — interaction type is "
            f"'{interaction_type}'. Only metadata will be saved."
        )
        # Save a minimal CSV with just the classification so the experiment
        # is still recorded and filterable in batch comparisons.
        rows = [
            {"Parameter": "interaction_type",    "Value": interaction_type,  "Model": "General", "Notes": ""},
            {"Parameter": "h_0_mm",              "Value": round(h0, 4),      "Model": "General", "Notes": ""},
            {"Parameter": "concentration_wv_pct","Value": concentration_wv_pct if concentration_wv_pct is not None else "N/A",
             "Model": "General", "Notes": "g per 100 mL"},
        ]
        csv_path = os.path.join(output_dir, "kinetic_parameters.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"  Saved minimal parameters CSV: {csv_path}")
        return None

    # ── Step 3: Concentration conversion ─────────────────────────────────────
    phi0 = None
    if concentration_wv_pct is not None and polymer_density_g_per_cm3 is not None:
        phi0 = wv_pct_to_volume_fraction(concentration_wv_pct, polymer_density_g_per_cm3)
        logger.info(f"  phi_0 (prep. volume fraction) = {phi0:.4f}")
    elif concentration_wv_pct is not None:
        logger.warning(
            "  concentration_wv_pct provided but polymer_density_g_per_cm3 is None — "
            "cannot compute phi_0. Pass polymer_density to enable conversion."
        )

    # ── Step 4: Fit kinetic models ────────────────────────────────────────────
    logger.info("\n  Fitting Schott PSO model...")
    pso_result = fit_schott_pso(time_s_clean, S)

    logger.info("  Fitting Korsmeyer-Peppas power law...")
    peppas_result = fit_korsmeyer_peppas(time_s_clean, S, Se)

    logger.info("  Fitting first-order baseline...")
    fo_result = fit_first_order(time_s_clean, S, Se)

    # ── Step 5: Flory-Rehner crosslink density ────────────────────────────────
    n_crosslink = None
    if chi is not None and pso_result["success"]:
        # Use the PSO equilibrium height (more robust than raw plateau mean)
        he_mm = h0 * (1.0 + pso_result["Se"])
        logger.info("\n  Computing Flory-Rehner crosslink density...")
        n_crosslink = compute_flory_rehner(
            h0_mm=h0,
            he_mm=he_mm,
            chi=chi,
            V1_m3_per_mol=V1_m3_per_mol
        )
    elif chi is not None:
        logger.warning(
            "  chi provided but PSO fit failed — cannot compute Flory-Rehner "
            "crosslink density (need PSO Se to determine he_mm)."
        )

    # ── Step 6: Log model comparison summary ─────────────────────────────────
    def _r2_str(result):
        return f"{result['r_squared']:.4f}" if result["success"] else "FAILED"

    logger.info("\n  ── Model Comparison Summary ──")
    logger.info(f"  {'Model':<22} {'R2':>8}  Key Parameters")
    logger.info(f"  {'-'*22}  {'-'*8}  {'-'*40}")
    if pso_result["success"]:
        logger.info(f"  {'Schott PSO':<22} {_r2_str(pso_result):>8}  "
                    f"Se={pso_result['Se']:.4f}, ks={pso_result['ks']:.6f}")
    else:
        logger.info(f"  {'Schott PSO':<22} {_r2_str(pso_result):>8}")
    if peppas_result["success"]:
        logger.info(f"  {'Korsmeyer-Peppas':<22} {_r2_str(peppas_result):>8}  "
                    f"n={peppas_result['n']:.4f}  ->  {peppas_result['mechanism']}")
    else:
        logger.info(f"  {'Korsmeyer-Peppas':<22} {_r2_str(peppas_result):>8}")
    if fo_result["success"]:
        logger.info(f"  {'First-Order':<22} {_r2_str(fo_result):>8}  "
                    f"k1={fo_result['k1']:.6f}")
    else:
        logger.info(f"  {'First-Order':<22} {_r2_str(fo_result):>8}")

    if pso_result["success"] and fo_result["success"]:
        if pso_result["r_squared"] > fo_result["r_squared"]:
            logger.info("  PSO > first-order -> stress-relaxation mechanism confirmed.")
        else:
            logger.info("  First-order > PSO — check data quality or consider "
                        "whether swelling is purely diffusion-driven.")

    if n_crosslink is not None:
        logger.info(f"  Flory-Rehner crosslink density: {n_crosslink:.4f} mol/m3")

    # ── Step 7: Save outputs ──────────────────────────────────────────────────
    try:
        csv_path, plot_path = save_kinetic_results(
            time_s=time_s_clean,
            S=S,
            Se=Se,
            h0=h0,
            pso=pso_result,
            peppas=peppas_result,
            first_order=fo_result,
            output_dir=output_dir,
            interaction_type=interaction_type,
            concentration_wv_pct=concentration_wv_pct,
            phi0=phi0,
            n_crosslink=n_crosslink,
            chi=chi,
        )
    except Exception as e:
        logger.error(f"Failed to save kinetic analysis outputs: {e}")
        return None

    logger.info("\n=== Kinetic Analysis Complete ===")
    logger.info(f"  Parameters CSV : {csv_path}")
    logger.info(f"  Plot           : {plot_path}")

    return csv_path, plot_path
