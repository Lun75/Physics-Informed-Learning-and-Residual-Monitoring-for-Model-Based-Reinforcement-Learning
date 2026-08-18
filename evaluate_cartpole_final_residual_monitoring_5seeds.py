"""
Final five-seed CartPole residual-monitoring evaluation.

Protocol
--------
1. Infer complete CartPole episodes from transition continuity.
2. Create disjoint episode-level calibration and test splits for each seed.
3. Estimate per-state-component observation-noise scales from nominal
   calibration targets only.
4. Add a fixed fractional Gaussian perturbation to nominal observed next
   states, smooth raw D_t with an EWMA, and calibrate a per-seed P95 threshold.
5. Freeze that threshold and observation-noise scale.
6. Evaluate clean nominal, benign noisy nominal, fine force-calibration, and
   rail-friction conditions.
7. Use one common test-noise realization for every noisy condition within a
   seed, so physical misspecification is the only changing factor.
8. Apply a 10-exceedances-in-20-transitions persistence rule within episodes.
9. Report mean +/- sample SD across five seeds.

The raw physical discrepancy is

    r_t = (s_obs_{t+1} - s_t) / dt - g_nominal(s_t, a_t)
    D_t = ||r_t||_2^2

Only the observed next-state endpoint receives the benign observation
perturbation. This preserves the convention used in the preceding CartPole
experiments. Describe it as "next-state observation noise", not as noise on
both endpoints of a physical sensor trajectory.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Experiment configuration

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# Local values reveal the transition between benign and detected force error;
# 0.95 and 1.05 retain clear stress-test endpoints.
FORCE_CALIBRATION_VALUES = [0.95, 0.99, 0.995, 0.998, 0.999, 0.9995, 1.0, 1.0005, 1.001, 1.002, 1.005, 1.01, 1.05]

RAIL_FRICTION_VALUES = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2]

DT = 0.02
EWMA_ALPHA = 0.10
THRESHOLD_QUANTILE = 0.95
PERSISTENCE_WINDOW = 20
PERSISTENCE_MIN_EXCEEDANCES = 10

# Noise is scaled independently for each state component using the nominal
# calibration-target SD. A value of 0.001 means 0.1% of that component's SD.
DEFAULT_NOISE_FRACTION = 0.001

CALIBRATION_RATIO = 0.15
TEST_RATIO = 0.15
EPISODE_CONTINUITY_TOL = 1e-6
PLUS_MINUS = "\N{PLUS-MINUS SIGN}"

SCRIPT_DIR = Path(__file__).resolve().parent


# Nominal CartPole physics and counterfactual targets


def cartpole_derivative_np(
    states: np.ndarray,
    actions: np.ndarray,
    force_calibration: float = 1.0,
) -> np.ndarray:
    """Return ds/dt for [x, x_dot, theta, theta_dot]."""
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0

    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)

    x_dot = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]

    force = (
        (2.0 * actions.astype(np.float64) - 1.0)
        * force_mag
        * force_calibration
    )

    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    temp = (
        force + polemass_length * theta_dot**2 * sintheta
    ) / total_mass

    theta_acc = (
        gravity * sintheta - costheta * temp
    ) / (
        length
        * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
    )

    x_acc = (
        temp - polemass_length * theta_acc * costheta / total_mass
    )

    return np.column_stack(
        [x_dot, x_acc, theta_dot, theta_acc]
    ).astype(np.float64)


def generate_force_targets(
    states: np.ndarray,
    actions: np.ndarray,
    force_calibration: float,
) -> np.ndarray:
    """Generate one-step explicit-Euler targets under F_env=c_F*F_nominal."""
    derivative = cartpole_derivative_np(
        states,
        actions,
        force_calibration=force_calibration,
    )
    return states.astype(np.float64) + DT * derivative


def generate_friction_targets(
    states: np.ndarray,
    nominal_targets: np.ndarray,
    beta_x: float,
) -> np.ndarray:
    """Apply the rail-friction convention used by the prior experiment."""
    targets = nominal_targets.astype(np.float64).copy()
    delta_x_dot = -beta_x * states[:, 1].astype(np.float64) * DT
    targets[:, 1] += delta_x_dot
    targets[:, 0] += DT * delta_x_dot
    return targets


#
# Episode-aware splitting
#

def infer_episode_ids(
    states: np.ndarray,
    next_states: np.ndarray,
    tolerance: float = EPISODE_CONTINUITY_TOL,
) -> np.ndarray:
    """Infer resets where next_states[i] is not states[i+1]."""
    n = len(states)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    continuity_error = np.max(
        np.abs(
            next_states[:-1].astype(np.float64)
            - states[1:].astype(np.float64)
        ),
        axis=1,
    )
    new_episode = continuity_error > tolerance

    episode_ids = np.zeros(n, dtype=np.int64)
    episode_ids[1:] = np.cumsum(new_episode)
    return episode_ids


def choose_episode_split(
    episode_ids: np.ndarray,
    seed: int,
    calibration_ratio: float = CALIBRATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose disjoint complete episodes for calibration and test."""
    unique_ids, counts = np.unique(episode_ids, return_counts=True)
    if len(unique_ids) < 2:
        raise RuntimeError("At least two complete episodes are required.")

    rng = np.random.default_rng(seed)
    shuffled_ids = unique_ids.copy()
    rng.shuffle(shuffled_ids)

    n_total = len(episode_ids)
    calibration_target = max(1, int(round(calibration_ratio * n_total)))
    test_target = max(1, int(round(test_ratio * n_total)))
    count_lookup = dict(zip(unique_ids, counts))

    calibration_episodes: list[int] = []
    test_episodes: list[int] = []
    calibration_count = 0
    test_count = 0
    cursor = 0

    while cursor < len(shuffled_ids) and calibration_count < calibration_target:
        episode_id = int(shuffled_ids[cursor])
        calibration_episodes.append(episode_id)
        calibration_count += int(count_lookup[episode_id])
        cursor += 1

    while cursor < len(shuffled_ids) and test_count < test_target:
        episode_id = int(shuffled_ids[cursor])
        test_episodes.append(episode_id)
        test_count += int(count_lookup[episode_id])
        cursor += 1

    if not calibration_episodes or not test_episodes:
        raise RuntimeError(
            "Could not create disjoint episode-level calibration/test splits."
        )

    calibration_indices = np.flatnonzero(
        np.isin(episode_ids, calibration_episodes)
    )
    test_indices = np.flatnonzero(np.isin(episode_ids, test_episodes))

    calibration_indices.sort()
    test_indices.sort()
    return calibration_indices, test_indices


#
# Residual monitor
#

def compute_D_t(
    states: np.ndarray,
    actions: np.ndarray,
    observed_next_states: np.ndarray,
) -> np.ndarray:
    """Compute raw observed-transition discrepancy D_t=||r_t||^2."""
    nominal_derivative = cartpole_derivative_np(
        states,
        actions,
        force_calibration=1.0,
    )
    observed_derivative = (
        observed_next_states.astype(np.float64)
        - states.astype(np.float64)
    ) / DT
    residual = observed_derivative - nominal_derivative
    return np.sum(residual**2, axis=1).astype(np.float64)


def apply_ewma_by_episode(
    values: np.ndarray,
    episode_ids: np.ndarray,
    alpha: float = EWMA_ALPHA,
) -> np.ndarray:
    """Apply EWMA independently within each episode."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EWMA alpha must lie in (0, 1].")

    values = np.asarray(values, dtype=np.float64)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    if len(values) != len(episode_ids):
        raise ValueError("values and episode_ids must have equal lengths.")

    smoothed = np.empty_like(values, dtype=np.float64)
    for episode_id in np.unique(episode_ids):
        idx = np.flatnonzero(episode_ids == episode_id)
        if len(idx) == 0:
            continue
        smoothed[idx[0]] = values[idx[0]]
        for position in range(1, len(idx)):
            current = idx[position]
            previous = idx[position - 1]
            smoothed[current] = (
                alpha * values[current]
                + (1.0 - alpha) * smoothed[previous]
            )
    return smoothed


def empirical_quantile(values: np.ndarray, quantile: float) -> float:
    """Return a conservative observed-sample quantile across NumPy versions."""
    try:
        return float(np.quantile(values, quantile, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, quantile, interpolation="higher"))


def compute_alert_metrics(
    d_t: np.ndarray,
    episode_ids: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate raw D_t summaries and EWMA-based ordinary/persistent alerts."""
    d_t = np.asarray(d_t, dtype=np.float64)
    d_bar_t = apply_ewma_by_episode(d_t, episode_ids, EWMA_ALPHA)
    alerts = d_bar_t > threshold

    persistent_flags: list[bool] = []
    for episode_id in np.unique(episode_ids):
        idx = np.flatnonzero(episode_ids == episode_id)
        episode_alerts = alerts[idx].astype(np.int64)
        if len(episode_alerts) < PERSISTENCE_WINDOW:
            continue

        rolling_counts = np.convolve(
            episode_alerts,
            np.ones(PERSISTENCE_WINDOW, dtype=np.int64),
            mode="valid",
        )
        persistent_flags.extend(
            (
                rolling_counts >= PERSISTENCE_MIN_EXCEEDANCES
            ).tolist()
        )

    eligible_windows = len(persistent_flags)
    persistent_rate = (
        float(100.0 * np.mean(np.asarray(persistent_flags, dtype=float)))
        if eligible_windows
        else float("nan")
    )

    return {
        "mean_D_t": float(np.mean(d_t)),
        "p95_D_t": float(np.quantile(d_t, 0.95)),
        "alert_rate_pct": float(100.0 * np.mean(alerts)),
        "persistent_alert_rate_pct": persistent_rate,
        "n_transitions": int(len(d_t)),
        "n_persistence_eligible_windows": int(eligible_windows),
    }


def calibrate_threshold(
    states: np.ndarray,
    actions: np.ndarray,
    nominal_next_states: np.ndarray,
    episode_ids: np.ndarray,
    noise_scale: np.ndarray,
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Calibrate one frozen P95 EWMA threshold using benign nominal noise."""
    rng = np.random.default_rng(seed + 100_000)
    calibration_noise = rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=nominal_next_states.shape,
    )
    benign_next_states = nominal_next_states.astype(np.float64) + calibration_noise

    benign_d_t = compute_D_t(states, actions, benign_next_states)
    benign_ewma = apply_ewma_by_episode(benign_d_t, episode_ids, EWMA_ALPHA)
    threshold = empirical_quantile(benign_ewma, THRESHOLD_QUANTILE)

    return threshold, {
        "calibration_mean_D_t": float(np.mean(benign_d_t)),
        "calibration_p95_D_t": float(np.quantile(benign_d_t, 0.95)),
        "calibration_mean_EWMA_D_t": float(np.mean(benign_ewma)),
        "calibration_p95_EWMA_D_t": threshold,
    }


#
# Aggregation and figures
#

REPORT_METRICS = [
    "mean_D_t",
    "p95_D_t",
    "alert_rate_pct",
    "persistent_alert_rate_pct",
]


def aggregate_results(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return numeric and formatted mean +/- sample-SD tables."""
    group_columns = ["experiment", "condition_value", "condition_label"]

    aggregate = (
        raw_df.groupby(group_columns, dropna=False)[REPORT_METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in aggregate.columns
    ]

    counts = (
        raw_df.groupby(group_columns, dropna=False, as_index=False)
        .agg(n_seeds=("seed", "nunique"))
    )
    aggregate = counts.merge(aggregate, on=group_columns, how="left")
    aggregate = aggregate.sort_values(
        ["experiment", "condition_value"],
        na_position="first",
    ).reset_index(drop=True)

    formatted = aggregate[group_columns + ["n_seeds"]].copy()
    for metric in REPORT_METRICS:
        formatted[f"{metric}_mean_std"] = aggregate.apply(
            lambda row: (
                f"{row[f'{metric}_mean']:.10g} {PLUS_MINUS} "
                f"{row[f'{metric}_std']:.3g}"
            ),
            axis=1,
        )
    return aggregate, formatted


def save_metric_plot(
    aggregate_df: pd.DataFrame,
    output_dir: Path,
    experiment: str,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:

    rows = aggregate_df[aggregate_df["experiment"] == experiment].sort_values("condition_value")
    if rows.empty:
        return

    x = rows["condition_value"].to_numpy(dtype=float)
    y = rows[f"{metric}_mean"].to_numpy(dtype=float)
    yerr = rows[f"{metric}_std"].to_numpy(dtype=float)

    plt.figure(figsize=(7.6, 4.8))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)

    if experiment == "force_calibration":
        plt.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        plt.xlabel("Force calibration $c_F$")

        if metric in [
            "alert_rate_pct",
            "persistent_alert_rate_pct",
        ]:
            plt.xlim(0.9945, 1.0055)
            plt.xticks([0.995, 0.998, 1.000, 1.002, 1.005])
        elif metric == "mean_D_t":
            plt.yscale("log")
    elif experiment == "rail_friction":
        plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        plt.xlabel(r"Rail friction $\beta_x$")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()

# Input validation and evaluation
def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Put cartpole_transitions.npz beside the "
            "script or provide --data."
        )

    with np.load(path) as data:
        required = {"states", "actions", "next_states"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing NPZ keys: {sorted(missing)}")
        states = data["states"].astype(np.float64)
        actions = data["actions"].astype(np.int64)
        next_states = data["next_states"].astype(np.float64)

    if not (len(states) == len(actions) == len(next_states)):
        raise ValueError("states, actions, and next_states must have equal lengths.")
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("states must have shape (N, 4).")
    if next_states.ndim != 2 or next_states.shape[1] != 4:
        raise ValueError("next_states must have shape (N, 4).")
    if not np.isfinite(states).all() or not np.isfinite(next_states).all():
        raise ValueError("states/next_states contain NaN or Inf.")
    if not np.isin(actions, [0, 1]).all():
        raise ValueError("CartPole actions must be 0 or 1.")
    return states, actions, next_states


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the final CartPole residual monitor."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SCRIPT_DIR / "cartpole_transitions.npz",
        help="Path to cartpole_transitions.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "cartpole_final_residual_monitoring_outputs",
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seeds to evaluate. Default: five dissertation seeds.",
    )
    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=DEFAULT_NOISE_FRACTION,
        help="Per-component benign noise as a fraction of calibration-target SD.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.noise_fraction <= 0:
        raise ValueError("--noise-fraction must be positive.")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates.")
    if not (0 < CALIBRATION_RATIO < 1 and 0 < TEST_RATIO < 1):
        raise ValueError("Calibration/test ratios must lie in (0, 1).")
    if CALIBRATION_RATIO + TEST_RATIO >= 1:
        raise ValueError("Calibration and test ratios must sum to less than 1.")
    if not (0 < PERSISTENCE_MIN_EXCEEDANCES <= PERSISTENCE_WINDOW):
        raise ValueError("Invalid persistence rule.")

    states, actions, next_states = load_dataset(args.data)
    episode_ids = infer_episode_ids(states, next_states)
    unique_episodes, episode_lengths = np.unique(
        episode_ids,
        return_counts=True,
    )
    if not np.any(episode_lengths >= PERSISTENCE_WINDOW):
        raise RuntimeError(
            "No inferred episode is long enough for the persistence window."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("FINAL CARTPOLE RESIDUAL MONITORING")
    print("=" * 78)
    print(f"Data: {args.data.resolve()}")
    print(f"Transitions: {len(states)}")
    print(f"Inferred episodes: {len(unique_episodes)}")
    print(
        "Episode length median/min/max: "
        f"{np.median(episode_lengths):.1f}/"
        f"{episode_lengths.min()}/{episode_lengths.max()}"
    )
    print(f"Seeds: {args.seeds}")
    print(f"Noise fraction: {args.noise_fraction:g}")
    print(f"EWMA alpha: {EWMA_ALPHA}")
    print(f"Threshold quantile: P{100 * THRESHOLD_QUANTILE:g}")
    print(
        "Persistence: "
        f"{PERSISTENCE_MIN_EXCEEDANCES}/{PERSISTENCE_WINDOW}"
    )

    all_results: list[dict] = []
    threshold_rows: list[dict] = []

    for seed in args.seeds:
        print("\n" + "-" * 20)
        print(f"Seed {seed}")
        print("-" * 20)

        calibration_indices, test_indices = choose_episode_split(
            episode_ids,
            seed,
        )
        state_cal = states[calibration_indices]
        action_cal = actions[calibration_indices]
        next_cal = next_states[calibration_indices]
        episode_cal = episode_ids[calibration_indices]

        state_test = states[test_indices]
        action_test = actions[test_indices]
        next_test_nominal = next_states[test_indices]
        episode_test = episode_ids[test_indices]

        # Estimate scale on calibration targets only; freeze it for the seed.
        calibration_target_std = np.std(next_cal, axis=0, ddof=0)
        noise_scale = args.noise_fraction * calibration_target_std
        noise_scale = np.maximum(noise_scale, np.finfo(np.float64).eps)

        threshold, calibration_summary = calibrate_threshold(
            state_cal,
            action_cal,
            next_cal,
            episode_cal,
            noise_scale,
            seed,
        )

        generated_nominal = generate_force_targets(
            state_test,
            action_test,
            force_calibration=1.0,
        )
        generation_max_error = float(
            np.max(np.abs(generated_nominal - next_test_nominal))
        )
        if generation_max_error > 1e-5:
            warnings.warn(
                "The explicit-Euler generator does not reproduce stored "
                f"nominal transitions (max error {generation_max_error:.3e})."
            )

        threshold_rows.append(
            {
                "seed": seed,
                "noise_fraction": args.noise_fraction,
                "noise_scale_x": noise_scale[0],
                "noise_scale_x_dot": noise_scale[1],
                "noise_scale_theta": noise_scale[2],
                "noise_scale_theta_dot": noise_scale[3],
                "ewma_alpha": EWMA_ALPHA,
                "threshold_quantile": THRESHOLD_QUANTILE,
                "alert_threshold": threshold,
                "persistence_window": PERSISTENCE_WINDOW,
                "persistence_min_exceedances": PERSISTENCE_MIN_EXCEEDANCES,
                "n_calibration_transitions": len(calibration_indices),
                "n_test_transitions": len(test_indices),
                "nominal_generation_max_abs_error": generation_max_error,
                **calibration_summary,
            }
        )

        print(f"Calibration transitions: {len(calibration_indices)}")
        print(f"Test transitions: {len(test_indices)}")
        print(f"Noise scale: {noise_scale}")
        print(f"Frozen threshold tau: {threshold:.10g}")

        common_fields = {
            "seed": seed,
            "noise_fraction": args.noise_fraction,
            "alert_threshold": threshold,
        }

        # A. Clean nominal sanity control (no observation noise).
        clean_d_t = compute_D_t(state_test, action_test, next_test_nominal)
        clean_metrics = compute_alert_metrics(
            clean_d_t,
            episode_test,
            threshold,
        )
        all_results.append(
            {
                **common_fields,
                "experiment": "nominal_control",
                "condition_value": 0.0,
                "condition_label": "clean_nominal",
                "observation_noise_applied": False,
                **clean_metrics,
            }
        )

        # Independent from calibration, but shared across every noisy test condition for paired/fair severity comparisons within the seed.
        monitoring_rng = np.random.default_rng(seed + 200_000)
        monitoring_noise = monitoring_rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=next_test_nominal.shape,
        )

        # B. Benign noisy nominal validation.
        benign_observed_targets = next_test_nominal + monitoring_noise
        benign_d_t = compute_D_t(
            state_test,
            action_test,
            benign_observed_targets,
        )
        benign_metrics = compute_alert_metrics(
            benign_d_t,
            episode_test,
            threshold,
        )
        all_results.append(
            {
                **common_fields,
                "experiment": "nominal_control",
                "condition_value": 1.0,
                "condition_label": "benign_noisy_nominal",
                "observation_noise_applied": True,
                **benign_metrics,
            }
        )

        # C. Force-calibration sweep with the same monitoring-noise draw.
        for c_f in FORCE_CALIBRATION_VALUES:
            clean_targets = (
                next_test_nominal.copy()
                if np.isclose(c_f, 1.0)
                else generate_force_targets(
                    state_test,
                    action_test,
                    force_calibration=c_f,
                )
            )
            observed_targets = clean_targets + monitoring_noise
            d_t = compute_D_t(state_test, action_test, observed_targets)
            metrics = compute_alert_metrics(d_t, episode_test, threshold)
            all_results.append(
                {
                    **common_fields,
                    "experiment": "force_calibration",
                    "condition_value": float(c_f),
                    "condition_label": f"c_F={c_f:g}",
                    "observation_noise_applied": True,
                    **metrics,
                }
            )
            print(
                f"force c_F={c_f:g}: "
                f"mean D={metrics['mean_D_t']:.6g}, "
                f"alert={metrics['alert_rate_pct']:.2f}%, "
                f"persistent={metrics['persistent_alert_rate_pct']:.2f}%"
            )

        # D. Rail-friction sweep with the same monitoring-noise draw.
        for beta_x in RAIL_FRICTION_VALUES:
            clean_targets = generate_friction_targets(
                state_test,
                next_test_nominal,
                beta_x,
            )
            observed_targets = clean_targets + monitoring_noise
            d_t = compute_D_t(state_test, action_test, observed_targets)
            metrics = compute_alert_metrics(d_t, episode_test, threshold)
            all_results.append(
                {
                    **common_fields,
                    "experiment": "rail_friction",
                    "condition_value": float(beta_x),
                    "condition_label": f"beta_x={beta_x:g}",
                    "observation_noise_applied": True,
                    **metrics,
                }
            )
            print(
                f"friction beta_x={beta_x:g}: "
                f"mean D={metrics['mean_D_t']:.6g}, "
                f"alert={metrics['alert_rate_pct']:.2f}%, "
                f"persistent={metrics['persistent_alert_rate_pct']:.2f}%"
            )

        pd.DataFrame(all_results).to_csv(
            output_dir / "cartpole_final_monitoring_raw_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(threshold_rows).to_csv(
            output_dir / "cartpole_final_monitoring_thresholds_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

    raw_df = pd.DataFrame(all_results)
    threshold_df = pd.DataFrame(threshold_rows)
    expected_rows = len(args.seeds) * (
        2 + len(FORCE_CALIBRATION_VALUES) + len(RAIL_FRICTION_VALUES)
    )
    if len(raw_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} raw rows, obtained {len(raw_df)}."
        )
    if raw_df[REPORT_METRICS].isna().any().any():
        raise RuntimeError(
            "A report metric is NaN. Check episode lengths and persistence eligibility."
        )

    aggregate_df, formatted_df = aggregate_results(raw_df)

    raw_path = output_dir / "cartpole_final_monitoring_raw.csv"
    aggregate_path = output_dir / "cartpole_final_monitoring_aggregate.csv"
    formatted_path = output_dir / "cartpole_final_monitoring_mean_std.csv"
    threshold_path = output_dir / "cartpole_final_monitoring_thresholds.csv"

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    formatted_df.to_csv(formatted_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    plot_specs = [
        ("mean_D_t", r"Mean $D_t$", "mean_D_t"),
        ("p95_D_t", r"P95 $D_t$", "p95_D_t"),
        ("alert_rate_pct", "Alert rate (%)", "alert_rate"),
        (
            "persistent_alert_rate_pct",
            "Persistent alert rate (%)",
            "persistent_alert_rate",
        ),
    ]

    for experiment, prefix in [
        ("force_calibration", "force"),
        ("rail_friction", "friction"),
    ]:
        for metric, ylabel, suffix in plot_specs:
            save_metric_plot(
                aggregate_df,
                output_dir,
                experiment,
                metric,
                ylabel,
                f"cartpole_{prefix}_{suffix}.png",
            )

    print("\n" + "=" * 20)
    print("FINAL MONITORING EVALUATION COMPLETE")
    print(f"Raw rows: {len(raw_df)}")
    print(f"Aggregate rows: {len(aggregate_df)}")
    print(f"Output directory: {output_dir}")
    print("Saved:")
    print(f"  {raw_path.name}")
    print(f"  {aggregate_path.name}")
    print(f"  {formatted_path.name}")
    print(f"  {threshold_path.name}")
    print("  eight PNG figures")

if __name__ == "__main__":
    main()
