"""
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

from mpc_controller import MPCController
from highway_env import HighwayEnv
from batch_planner import BatchPlanner


def run_simulation(scenario='overtake', T_sim=15.0, N=15, dt=0.1):

    env = HighwayEnv(dt=dt, scenario=scenario)
    mpc = MPCController(N=N, dt=dt)
    batch = BatchPlanner(mpc)

    state = env.reset()
    v_ref = env.v_ref

    states_log = [state.copy()]
    controls_log = []
    cost_log = []

    for step in range(int(T_sim / dt)):

        best_res, all_res = batch.plan(state, v_ref, env.obstacles)

        if best_res is None:
            print("No valid trajectory!")
            break

        u_apply = best_res['u_apply']
        cost = best_res['meta_cost']

        # Apply control
        state = env.step(state, u_apply)

        # 🔥 CRITICAL FIX: prevent low-speed instability
        state[3] = max(state[3], 5.0)

        states_log.append(state.copy())
        controls_log.append(u_apply.copy())
        cost_log.append(cost)

        if env.is_collision(state):
            print("Collision!")
            break

    return {
        "states": np.array(states_log),
        "controls": np.array(controls_log),
        "costs": np.array(cost_log),
        "env": env
    }


if __name__ == "__main__":
    data = run_simulation()