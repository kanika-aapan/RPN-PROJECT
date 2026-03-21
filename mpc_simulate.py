"""
mpc_simulate.py
===============
Full closed-loop MPC simulation on the highway environment.

Usage
-----
  python3 mpc_simulate.py --scenario cruise   --T 15
  python3 mpc_simulate.py --scenario overtake --T 20
  python3 mpc_simulate.py --scenario dense    --T 25

_select_y_ref — v8 (final)
---------------------------
v7 collision at step 125 root cause:
  At step 108: current lane y=4 had TTC=1.35s < MIN_TTC=1.5s.
  Escape mode fired and switched to y=0 (TTC=2.12s >= MIN_TTC=1.5s).
  But y=0 had a slow car (vx=9.56) that ego (v=19.6) closed on rapidly.
  After switch, ego targeted y=0, approached the slow car, and collided.

  The escape candidate had TTC=2.12s — enough to satisfy the old gate,
  but NOT enough for ego to safely execute the lane change AND decelerate.
  A 4m lane change at v=19.6 takes ~1.5s of steering. By the time ego
  reached y=0, the y=0 obstacle TTC had dropped below the collision threshold.

v8 fixes:
  1. MIN_TTC_ESCAPE = 3.0s: escape candidates must have at least 3s TTC,
     giving enough margin to complete the lane change and adjust speed.
     Normal-mode candidates still only need MIN_TTC=1.5s.
     Verified: step 108 y=0 TTC=2.12 < 3.0 → no escape → MPC brakes ✓

  2. COOLDOWN = 20 steps (2s): after any lane change, no further switch
     is allowed for 20 steps. This prevents re-switching while mid-manoeuvre
     and oscillating between lanes that are both marginal.
"""

import argparse
import time
import os
import numpy as np

from vehicle_model import (rollout, V_MIN, V_MAX,
                           A_MIN, A_MAX, DELTA_MIN, DELTA_MAX)
from mpc_controller import MPCController
from highway_env   import HighwayEnv, LANE_CENTERS, LANE_WIDTH, OBS_RADIUS


# ── lane selector parameters ──────────────────────────────────────────────────
_LANE_TOL        = LANE_WIDTH * 0.6
_REAR_MARGIN     = 5.0                # [m] beside check behind ego
_FORWARD_MARGIN  = 10.0               # [m] beside check ahead of ego
_MAX_LANE_JUMP   = LANE_WIDTH * 1.5
_MIN_TTC         = 1.5                # [s] normal-mode minimum TTC
_MIN_TTC_ESCAPE  = 3.0                # [s] escape-mode minimum TTC (v8)
_LC_COEFF        = 0.2
_HYSTERESIS      = 1.0
_COOLDOWN_STEPS  = 20                 # [steps] lock after lane change (v8)


def _make_warm_start(state0, v_ref, y_target, N, dt):
    v_cur = state0[3]; y_cur = state0[1]; u0 = np.zeros((N, 2))
    for k in range(N):
        a_k = np.clip((v_ref - v_cur) / dt, A_MIN, A_MAX)
        u0[k, 0] = a_k
        dy = (y_target - y_cur) / max(1, N - k)
        v_s = max(v_cur, 1.0)
        u0[k, 1] = np.clip(dy / (v_s * dt), DELTA_MIN, DELTA_MAX)
        v_cur = np.clip(v_cur + a_k * dt, V_MIN, V_MAX)
        y_cur += v_s * u0[k, 1] * dt
    return u0.flatten()


def _lane_ttc(lane_y, ego_x, ego_v, obstacles):
    """Asymmetric beside check: rear=5m, forward=10m. Sentinel=999."""
    min_ttc = 999.0; bb = False
    for obs in obstacles:
        px, py = obs.predict(0.0)
        if abs(py - lane_y) > _LANE_TOL: continue
        gap = px - ego_x
        if -_REAR_MARGIN < gap < _FORWARD_MARGIN: bb = True
        if gap > 0:
            cl = ego_v - obs.vx
            ttc = gap / cl if cl > 0 else gap / max(ego_v, 1.0)
            min_ttc = min(min_ttc, ttc)
    return min_ttc, bb


def _select_y_ref(ego_state, obstacles, current_y_ref, cooldown):
    """
    Returns (new_y_ref, new_cooldown).

    During cooldown: no switch allowed.
    Escape mode (current_ttc < MIN_TTC): switch to best lane with
      TTC >= MIN_TTC_ESCAPE; if none, stay (MPC brakes).
    Normal mode: switch if TTC advantage > HYSTERESIS.
    Any switch starts COOLDOWN_STEPS cooldown.
    """
    if cooldown > 0:
        return current_y_ref, cooldown - 1

    ego_x = ego_state[0]; ego_y = ego_state[1]
    ego_v = max(float(ego_state[3]), 1.0)

    current_ttc, _ = _lane_ttc(current_y_ref, ego_x, ego_v, obstacles)

    # collect candidate lanes (beside-safe)
    candidates = []
    for lane_y in LANE_CENTERS:
        if lane_y == current_y_ref: continue
        if abs(lane_y - ego_y) > _MAX_LANE_JUMP: continue
        ttc, bb = _lane_ttc(lane_y, ego_x, ego_v, obstacles)
        if bb: continue
        score = ttc - abs(lane_y - ego_y) * _LC_COEFF
        candidates.append((score, ttc, lane_y))

    if not candidates:
        return current_y_ref, 0

    best_score, best_ttc, best_y = max(candidates)

    # escape mode: current lane critically dangerous
    if current_ttc < _MIN_TTC:
        # only escape if destination has enough margin for lane-change + deceleration
        safe = [(s, t, l) for s, t, l in candidates if t >= _MIN_TTC_ESCAPE]
        if safe:
            _, _, escape_y = max(safe)
            return escape_y, _COOLDOWN_STEPS
        return current_y_ref, 0   # no safe escape — stay, MPC brakes

    # normal mode
    if best_ttc >= _MIN_TTC and best_score > current_ttc + _HYSTERESIS:
        return best_y, _COOLDOWN_STEPS

    return current_y_ref, 0


def run_simulation(scenario='cruise', T_sim=15.0, N=15, dt=0.1, verbose=True):
    env   = HighwayEnv(dt=dt, scenario=scenario)
    mpc   = MPCController(N=N, dt=dt)
    state = env.reset()
    v_ref = env.v_ref
    y_ref = env.get_lane_center()

    u_init   = _make_warm_start(state, v_ref, y_ref, N, dt)
    cooldown = 0

    states_log=[state.copy()]; controls_log=[]; cost_log=[]
    solve_t_log=[]; predicted_trajs=[]; y_ref_log=[y_ref]

    n_steps=int(T_sim/dt); collision=False

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Scenario : {scenario.upper()}")
        print(f"  Horizon  : N={N}, dt={dt}s")
        print(f"  v_ref    : {v_ref} m/s  |  y_ref : {y_ref} m (adaptive)")
        print(f"{'='*55}")
        print(f"{'Step':>5} | {'x':>7} | {'y':>6} | {'v':>6} | "
              f"{'y_ref':>6} | {'cost':>9} | {'t_solve':>8}")
        print("-"*65)

    for step in range(n_steps):
        obs_list = env.obstacles

        y_ref, cooldown = _select_y_ref(state, obs_list, y_ref, cooldown)

        t0 = time.perf_counter()
        res = mpc.solve(state, y_ref=y_ref, v_ref=v_ref,
                        obstacles=obs_list, u_init=u_init)
        dt_solve = time.perf_counter() - t0

        u_init  = None
        u_apply = res['u_apply']
        cost    = res['cost']
        state   = env.step(state, u_apply)

        states_log.append(state.copy())
        controls_log.append(u_apply.copy())
        cost_log.append(cost)
        solve_t_log.append(dt_solve)
        predicted_trajs.append(res['trajectory'].copy())
        y_ref_log.append(y_ref)

        if verbose and step % 10 == 0:
            print(f"{step:>5} | {state[0]:>7.1f} | {state[1]:>6.2f} | "
                  f"{state[3]:>6.2f} | {y_ref:>6.2f} | "
                  f"{cost:>9.2f} | {dt_solve:>7.4f}s")

        if env.is_collision(state):
            print(f"\n⚠  Collision at step {step}!")
            collision = True
            break

    states_arr   = np.array(states_log)
    controls_arr = np.array(controls_log)
    cost_arr     = np.array(cost_log)
    solve_arr    = np.array(solve_t_log)

    if verbose:
        dy_arr = np.abs(states_arr[:,1] - np.array(y_ref_log[:len(states_arr)]))
        print(f"\n── Summary ────────────────────────────────────────────")
        print(f"  Mean speed      : {states_arr[:,3].mean():.2f} m/s")
        print(f"  Speed range     : [{states_arr[:,3].min():.2f}, {states_arr[:,3].max():.2f}] m/s")
        print(f"  Mean |y-y_ref|  : {dy_arr.mean():.3f} m")
        print(f"  Max  |y-y_ref|  : {dy_arr.max():.3f} m")
        print(f"  Mean cost       : {cost_arr.mean():.2f}")
        print(f"  Solve time      : {solve_arr.mean()*1000:.1f} ms ± {solve_arr.std()*1000:.1f} ms")
        print(f"  Max solve time  : {solve_arr.max()*1000:.1f} ms")
        print(f"  Collisions      : {'Yes' if collision else 'No'}")
        print(f"{'='*55}\n")

    return dict(states=states_arr, controls=controls_arr, costs=cost_arr,
                solve_times=solve_arr, predicted_trajs=predicted_trajs,
                scenario=scenario, env=env, v_ref=v_ref, y_ref=y_ref,
                dt=dt, N=N, collision=collision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="cruise", choices=["cruise","overtake","dense"])
    parser.add_argument("--T",  type=float, default=15.0)
    parser.add_argument("--N",  type=int,   default=15)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    data = run_simulation(scenario=args.scenario, T_sim=args.T, N=args.N, dt=args.dt)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    np.savez(os.path.join(out_dir, f"trajectory_{args.scenario}.npz"),
             **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
    print(f"Data saved → trajectory_{args.scenario}.npz")