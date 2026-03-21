"""
mpc_controller.py
=================
Model Predictive Controller for highway driving.

FIXES from original + bilateral road-edge barrier
--------------------------------------------------
1. k=0 Δa/Δδ uses last applied control (not hardcoded 0).
2. Δa/Δδ look-ahead gradient sign verified correct.
3. Obstacle barrier: quadratic inside safe_dist=3.5m, w_obs=50.
4. Analytical Jacobians from vehicle_model.linearize().
5. Smooth heading penalty: quadratic beyond ±30°.
6. ftol = 1e-4.
7. BILATERAL road-edge barrier (this version):
     Original: y_excess = abs(yk) - 9.5  → fires only when |y| > 9.5.
               y=-1.17 gives abs(-1.17)=1.17 < 9.5 → NO PENALTY.
               Obstacle barrier pushed ego to y=-1.17 (off left road edge)
               with no resistance → collision.
     Fixed:    Two separate one-sided barriers:
                 LEFT:  yk < Y_MIN = -0.5  → J += w_lane*(Y_MIN-yk)^2
                 RIGHT: yk > Y_MAX =  9.5  → J += w_lane*(yk-Y_MAX)^2
               w_lane raised from 80 → 500 so the barrier is strong enough
               to compete with the obstacle repulsion (w_obs=50 at d=3m → 528).
               At y=-0.8 (excess=0.3): lane_cost=500*0.09=45, growing fast.
               At y=-1.17 (excess=0.67): lane_cost=500*0.45=224 — strong.
     Gradient: dJ/d(yk) for J=w*(Y_MIN-yk)^2 is -2w*(Y_MIN-yk) = -2w*excess < 0
               Negative gradient → J decreases as yk increases → optimizer
               raises yk back into the road. Accumulated in dJ_dstates[k,1].
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from vehicle_model import (
    bicycle_dynamics, linearize, rollout,
    A_MIN, A_MAX, DELTA_MIN, DELTA_MAX,
)

DEFAULT_WEIGHTS = dict(
    w_y    = 1.0,
    w_v    = 2.0,
    w_th   = 0.1,
    w_a    = 0.5,
    w_d    = 1.0,
    w_da   = 2.0,
    w_dd   = 5.0,
    w_yN   = 10.0,
    w_vN   = 10.0,
    w_obs  = 50.0,
    w_lane = 500.0,    # raised from 80 — must dominate obs barrier at road edge
)

_SAFE_DIST   = 3.5            # [m] matches OBS_RADIUS in highway_env.py
_Y_MIN       = -0.5           # [m] left road boundary (lane 0 centre=0, width=4 → edge=-2, margin=1.5)
_Y_MAX       =  9.5           # [m] right road boundary (lane 8 centre=8, width=4 → edge=10, margin=0.5)
_THETA_LIMIT = np.pi / 6.0   # [rad] 30° smooth heading penalty onset


class MPCController:

    def __init__(self, N=15, dt=0.1, weights=None):
        self.N  = N
        self.dt = dt
        self.w  = {**DEFAULT_WEIGHTS, **(weights or {})}

        lb = np.tile([A_MIN, DELTA_MIN], N)
        ub = np.tile([A_MAX, DELTA_MAX], N)
        self.bounds = Bounds(lb, ub)

        self._u_prev    = np.zeros(N * 2)
        self._a_applied = 0.0
        self._d_applied = 0.0

    def solve(self, state0, y_ref, v_ref, obstacles=None, u_init=None):
        obstacles = obstacles or []
        u0 = u_init if u_init is not None else self._u_prev.copy()

        res = minimize(
            fun    = self._cost_and_grad,
            x0     = u0,
            jac    = True,
            method = "L-BFGS-B",
            bounds = self.bounds,
            args   = (state0, y_ref, v_ref, obstacles,
                      self._a_applied, self._d_applied),
            options = dict(maxiter=200, ftol=1e-4, gtol=1e-5),
        )

        u_opt = res.x.reshape(self.N, 2)
        traj  = rollout(state0, u_opt, self.dt)

        self._u_prev    = np.vstack([u_opt[1:], u_opt[-1]]).flatten()
        self._a_applied = float(u_opt[0, 0])
        self._d_applied = float(u_opt[0, 1])

        return dict(
            controls   = u_opt,
            trajectory = traj,
            cost       = res.fun,
            u_apply    = u_opt[0],
            success    = res.success,
        )

    def _cost_and_grad(self, u_flat, state0, y_ref, v_ref, obstacles,
                       a_applied, d_applied):
        w  = self.w
        N  = self.N
        dt = self.dt
        u  = u_flat.reshape(N, 2)

        states = np.zeros((N + 1, 4))
        states[0] = state0
        for k in range(N):
            states[k + 1] = bicycle_dynamics(states[k], u[k], dt)

        J          = 0.0
        dJ_dstates = np.zeros((N + 1, 4))
        dJ_du      = np.zeros((N, 2))

        theta_ref = 0.0

        for k in range(N):
            xk, yk, thk, vk = states[k]
            ak, dk = u[k]

            ak_prev = a_applied if k == 0 else u[k - 1, 0]
            dk_prev = d_applied if k == 0 else u[k - 1, 1]

            # stage costs
            J += w['w_y']  * (yk - y_ref) ** 2
            J += w['w_v']  * (vk - v_ref) ** 2
            J += w['w_th'] * (thk - theta_ref) ** 2
            J += w['w_a']  * ak ** 2
            J += w['w_d']  * dk ** 2
            J += w['w_da'] * (ak - ak_prev) ** 2
            J += w['w_dd'] * (dk - dk_prev) ** 2

            # smooth heading penalty
            th_excess = abs(thk) - _THETA_LIMIT
            if th_excess > 0:
                J += 20.0 * th_excess ** 2

            # obstacle barrier
            for obs in obstacles:
                ox, oy = obs.predict(k * dt)
                dx, dy = xk - ox, yk - oy
                dist2  = dx * dx + dy * dy
                margin = _SAFE_DIST ** 2 - dist2
                if margin > 0:
                    J += w['w_obs'] * margin ** 2
                    dJ_dstates[k, 0] -= 4 * w['w_obs'] * margin * dx
                    dJ_dstates[k, 1] -= 4 * w['w_obs'] * margin * dy

            # BILATERAL road-edge barrier (FIX 7)
            # LEFT boundary: penalise yk < Y_MIN
            if yk < _Y_MIN:
                excess_l = _Y_MIN - yk          # positive
                J += w['w_lane'] * excess_l ** 2
                # dJ/d(yk) = -2*w*(Y_MIN-yk) = -2*w*excess_l  (negative)
                dJ_dstates[k, 1] -= 2 * w['w_lane'] * excess_l

            # RIGHT boundary: penalise yk > Y_MAX
            elif yk > _Y_MAX:
                excess_r = yk - _Y_MAX          # positive
                J += w['w_lane'] * excess_r ** 2
                # dJ/d(yk) = +2*w*(yk-Y_MAX) = +2*w*excess_r  (positive)
                dJ_dstates[k, 1] += 2 * w['w_lane'] * excess_r

            # state gradients
            dJ_dstates[k, 1] += 2 * w['w_y']  * (yk - y_ref)
            dJ_dstates[k, 2] += 2 * w['w_th'] * (thk - theta_ref)
            dJ_dstates[k, 3] += 2 * w['w_v']  * (vk - v_ref)

            if th_excess > 0:
                sign_th = 1.0 if thk >= 0 else -1.0
                dJ_dstates[k, 2] += 40.0 * th_excess * sign_th

            # control gradients
            dJ_du[k, 0] += 2 * w['w_a']  * ak
            dJ_du[k, 1] += 2 * w['w_d']  * dk
            dJ_du[k, 0] += 2 * w['w_da'] * (ak - ak_prev)
            dJ_du[k, 1] += 2 * w['w_dd'] * (dk - dk_prev)

            if k + 1 < N:
                dJ_du[k, 0] -= 2 * w['w_da'] * (u[k + 1, 0] - ak)
                dJ_du[k, 1] -= 2 * w['w_dd'] * (u[k + 1, 1] - dk)

        # terminal cost
        xN, yN, _, vN = states[N]
        J += w['w_yN'] * (yN - y_ref) ** 2
        J += w['w_vN'] * (vN - v_ref) ** 2

        dJ_dstates[N, 1] += 2 * w['w_yN'] * (yN - y_ref)
        dJ_dstates[N, 3] += 2 * w['w_vN'] * (vN - v_ref)

        # backward pass: analytical Jacobians
        lam = dJ_dstates[N].copy()
        for k in range(N - 1, -1, -1):
            A, B = linearize(states[k], u[k], dt)
            dJ_du[k] += B.T @ lam
            lam       = A.T @ lam + dJ_dstates[k]

        return J, dJ_du.flatten()