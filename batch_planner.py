import numpy as np
from highway_env import LANE_CENTERS


class BatchPlanner:
    def __init__(self, mpc_controller):

        self.mpc = mpc_controller

        # FINAL tuned weights
        self.w_lane = 12.0
        self.w_speed = 2.0
        self.w_collision = 50.0
        self.w_smooth = 2.0
        self.w_lane_change = 10.0
        self.w_progress = 1.0

        self.prev_best_lane = None

    def generate_candidates(self, current_y):
        return [
            lane_y for lane_y in LANE_CENTERS
            if abs(lane_y - current_y) <= 4.5
        ]

    def compute_meta_cost(self, result, y_ref, current_y, obstacles):

        traj = result['trajectory']
        controls = result['controls']

        cost = 0.0

        # =========================
        # 1. Lane tracking (strong)
        # =========================
        cost += self.w_lane * np.mean((traj[:, 1] - y_ref) ** 2)

        # =========================
        # 2. Speed tracking
        # =========================
        cost += self.w_speed * np.mean((traj[:, 3] - 20.0) ** 2)

        # =========================
        # 3. Lane change penalty
        # =========================
        cost += self.w_lane_change * abs(y_ref - current_y)

        # =========================
        # 4. Forward progress
        # =========================
        progress = traj[-1, 0] - traj[0, 0]
        cost -= self.w_progress * progress

        # =========================
        # 5. Collision penalty
        # =========================
        for k, state in enumerate(traj):
            px, py = state[0], state[1]
            for obs in obstacles:
                ox, oy = obs.predict(k * 0.1)
                dist = np.hypot(px - ox, py - oy)

                if dist < 8.0:
                    cost += self.w_collision * (8.0 - dist) ** 2

        # =========================
        # 6. Smoothness
        # =========================
        for k in range(1, len(controls)):
            cost += self.w_smooth * np.linalg.norm(
                controls[k] - controls[k - 1]
            )

        # =========================
        # 7. HARD + BUFFER BOUNDARY
        # =========================
        LOW = -0.5
        HIGH = 9.5
        BUFFER = 0.5

        for state in traj:
            y = state[1]

            # LEFT SIDE
            if y < LOW + BUFFER:
                cost += 1500.0 * (LOW + BUFFER - y) ** 2

            # RIGHT SIDE
            elif y > HIGH - BUFFER:
                cost += 1500.0 * (y - (HIGH - BUFFER)) ** 2

        return cost

    def plan(self, state, v_ref, obstacles):

        current_y = state[1]
        candidates = self.generate_candidates(current_y)

        best_cost = np.inf
        best_result = None
        best_lane = None
        all_results = []

        for lane_y in candidates:

            res = self.mpc.solve(
                state,
                y_ref=lane_y,
                v_ref=v_ref,
                obstacles=obstacles
            )

            if not res['success']:
                continue

            traj = res['trajectory']

            # Reject unstable trajectories
            if np.any(np.abs(traj[:, 2]) > 1.0):
                continue

            meta_cost = self.compute_meta_cost(
                res, lane_y, current_y, obstacles
            )

            res['meta_cost'] = meta_cost
            res['lane'] = lane_y

            all_results.append(res)

            if meta_cost < best_cost:
                best_cost = meta_cost
                best_result = res
                best_lane = lane_y

        if best_result is None:
            return None, all_results

        # =========================
        # Lane commitment
        # =========================
        if self.prev_best_lane is not None and best_lane is not None:
            if abs(best_lane - self.prev_best_lane) > 2.0:
                for res in all_results:
                    if abs(res['lane'] - self.prev_best_lane) < 1.0:
                        best_result = res
                        best_lane = self.prev_best_lane
                        break

        self.prev_best_lane = best_lane

        return best_result, all_results