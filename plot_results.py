"""
plot_results.py
===============
Publication-quality plots for the MPC highway simulation.

Generates:
  1. Top-view trajectory plot (ego + obstacles + lane markings)
  2. Control outputs (acceleration + steering over time)
  3. Speed profile vs reference
  4. MPC cost over time
  5. Solve-time histogram
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from highway_env import LANE_CENTERS, LANE_WIDTH, OBS_LENGTH, OBS_WIDTH


# ── colour palette ────────────────────────────────────────────────────────────
EGO_COLOR  = "#E63946"    # vivid red
OBS_COLOR  = "#457B9D"    # steel blue
LANE_COLOR = "#A8DADC"
REF_COLOR  = "#2A9D8F"
PRED_COLOR = "#F4A261"


def _lane_background(ax, x_min, x_max):
    """Draw lane markings on an axis."""
    road_y_min = LANE_CENTERS[0]  - LANE_WIDTH / 2
    road_y_max = LANE_CENTERS[-1] + LANE_WIDTH / 2
    ax.add_patch(patches.Rectangle(
        (x_min, road_y_min),
        x_max - x_min, road_y_max - road_y_min,
        facecolor="#1A1A2E", alpha=0.9, zorder=0
    ))
    for lc in LANE_CENTERS:
        ax.axhline(lc, color="#FFD166", lw=0.6, ls="--", alpha=0.5, zorder=1)
    ax.axhline(road_y_min, color="white", lw=1.5, zorder=1)
    ax.axhline(road_y_max, color="white", lw=1.5, zorder=1)


def plot_trajectory(data: dict, save_path: str = None):
    """Bird's-eye view of ego trajectory with obstacles."""
    states   = data['states']
    env      = data['env']
    y_ref    = data['y_ref']
    preds    = data['predicted_trajs']

    xs = states[:, 0]
    x_min, x_max = xs[0] - 5, xs[-1] + 20

    fig, ax = plt.subplots(figsize=(16, 5), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    _lane_background(ax, x_min, x_max)

    # lane centre reference
    ax.axhline(y_ref, color=REF_COLOR, lw=1.2, ls="-.", alpha=0.7,
               label="Lane centre", zorder=2)

    # predicted horizons (sparse)
    step_interval = max(1, len(preds) // 12)
    for i, traj in enumerate(preds):
        if i % step_interval == 0:
            ax.plot(traj[:, 0], traj[:, 1],
                    color=PRED_COLOR, lw=0.8, alpha=0.35, zorder=3)

    # obstacle initial positions
    for obs in env.obstacles:
        rect = patches.FancyBboxPatch(
            (obs.x - OBS_LENGTH / 2, obs.y - OBS_WIDTH / 2),
            OBS_LENGTH, OBS_WIDTH,
            boxstyle="round,pad=0.3",
            linewidth=1.2, edgecolor="white", facecolor=OBS_COLOR,
            alpha=0.85, zorder=4
        )
        ax.add_patch(rect)
        ax.text(obs.x, obs.y, f"{obs.vx:.0f}", color="white",
                fontsize=7, ha='center', va='center', zorder=5)

    # ego trajectory
    ax.plot(states[:, 0], states[:, 1], color=EGO_COLOR, lw=2.5,
            label="Ego trajectory", zorder=6)
    ax.scatter(states[0, 0], states[0, 1], color="white", s=80,
               zorder=7, label="Start")
    ax.scatter(states[-1, 0], states[-1, 1], color=EGO_COLOR,
               s=80, marker="*", zorder=7, label="End")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(LANE_CENTERS[0] - LANE_WIDTH, LANE_CENTERS[-1] + LANE_WIDTH)
    ax.set_xlabel("x [m]", color="white", fontsize=11)
    ax.set_ylabel("y [m]", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    scenario = data['scenario'].upper()
    ax.set_title(f"Batch MPC – {scenario} Scenario  |  Ego Trajectory",
                 color="white", fontsize=13, pad=10)
    leg = ax.legend(facecolor="#1F2937", edgecolor="#444",
                    labelcolor="white", fontsize=9, loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Saved: {save_path}")
    return fig


def plot_controls(data: dict, save_path: str = None):
    """Acceleration and steering angle over time."""
    controls = data['controls']
    dt       = data['dt']
    t        = np.arange(len(controls)) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                   facecolor="#0D1117", sharex=True)
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    ax1.plot(t, controls[:, 0], color=EGO_COLOR, lw=1.8, label="Acceleration")
    ax1.axhline(0, color="#555", lw=0.8, ls="--")
    ax1.fill_between(t, controls[:, 0], alpha=0.15, color=EGO_COLOR)
    ax1.set_ylabel("Acceleration [m/s²]", fontsize=10)
    ax1.legend(facecolor="#1F2937", edgecolor="#444",
               labelcolor="white", fontsize=9)
    ax1.set_title("Control Outputs", color="white", fontsize=12, pad=8)

    ax2.plot(t, np.degrees(controls[:, 1]), color="#06D6A0",
             lw=1.8, label="Steering angle")
    ax2.axhline(0, color="#555", lw=0.8, ls="--")
    ax2.fill_between(t, np.degrees(controls[:, 1]), alpha=0.15, color="#06D6A0")
    ax2.set_ylabel("Steering [deg]", fontsize=10)
    ax2.set_xlabel("Time [s]", color="white", fontsize=10)
    ax2.legend(facecolor="#1F2937", edgecolor="#444",
               labelcolor="white", fontsize=9)
    ax2.tick_params(axis='x', colors="white")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Saved: {save_path}")
    return fig


def plot_speed_and_cost(data: dict, save_path: str = None):
    """Speed profile + MPC cost side-by-side."""
    states = data['states']
    costs  = data['costs']
    dt     = data['dt']
    v_ref  = data['v_ref']
    t_s    = np.arange(len(states)) * dt
    t_c    = np.arange(len(costs))  * dt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                   facecolor="#0D1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # speed
    ax1.plot(t_s, states[:, 3], color="#F4A261", lw=2, label="Ego speed")
    ax1.axhline(v_ref, color=REF_COLOR, lw=1.5, ls="--", label=f"v_ref={v_ref} m/s")
    ax1.fill_between(t_s, states[:, 3], v_ref, alpha=0.1, color="#F4A261")
    ax1.set_xlabel("Time [s]", color="white"); ax1.set_ylabel("Speed [m/s]", color="white")
    ax1.set_title("Speed Profile", color="white", fontsize=12)
    ax1.legend(facecolor="#1F2937", edgecolor="#444", labelcolor="white", fontsize=9)
    ax1.tick_params(axis='both', colors="white")
    ax1.yaxis.label.set_color("white")

    # cost
    ax2.semilogy(t_c, costs, color="#A8DADC", lw=1.5)
    ax2.fill_between(t_c, costs, alpha=0.1, color="#A8DADC")
    ax2.set_xlabel("Time [s]", color="white"); ax2.set_ylabel("MPC Cost (log)", color="white")
    ax2.set_title("MPC Objective Cost over Time", color="white", fontsize=12)
    ax2.tick_params(axis='both', colors="white")
    ax2.yaxis.label.set_color("white")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Saved: {save_path}")
    return fig


def plot_solve_times(data: dict, save_path: str = None):
    """Histogram of MPC solve times per step."""
    ts = data['solve_times'] * 1000   # ms
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0D1117")
    ax.set_facecolor("#111827")
    ax.hist(ts, bins=25, color=EGO_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(ts.mean(), color="white", lw=1.5, ls="--",
               label=f"Mean={ts.mean():.1f} ms")
    ax.axvline(ts.max(),  color=PRED_COLOR, lw=1.5, ls=":",
               label=f"Max={ts.max():.1f} ms")
    ax.set_xlabel("Solve time [ms]", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("MPC Solve Time Distribution", color="white", fontsize=12)
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#1F2937", edgecolor="#444",
              labelcolor="white", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  → Saved: {save_path}")
    return fig


def plot_all(data: dict, prefix: str = "mpc"):
    """Generate and save all plots."""
    scen = data['scenario']
    print(f"\n── Generating plots for scenario '{scen}' ──")
    plot_trajectory   (data, f"{prefix}_{scen}_trajectory.png")
    plot_controls     (data, f"{prefix}_{scen}_controls.png")
    plot_speed_and_cost(data, f"{prefix}_{scen}_speed_cost.png")
    plot_solve_times  (data, f"{prefix}_{scen}_solve_times.png")
    print("── All plots saved. ──\n")