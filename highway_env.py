
import numpy as np


LANE_WIDTH   = 4.0
LANE_CENTERS = [0.0, 4.0, 8.0]
N_LANES      = 3

OBS_LENGTH   = 4.5
OBS_WIDTH    = 2.0
OBS_RADIUS   = 3.5   # used by MPC as _SAFE_DIST


class ObstacleVehicle:
    """Constant-velocity obstacle on the highway."""

    def __init__(self, x: float, y: float, vx: float, vy: float = 0.0):
        self.x  = x
        self.y  = y
        self.vx = vx
        self.vy = vy

    def step(self, dt: float):
        self.x += self.vx * dt
        self.y += self.vy * dt

    @property
    def state(self):
        return np.array([self.x, self.y, self.vx, self.vy])

    def predict(self, t: float):
        """Predicted position at time t seconds from now."""
        return self.x + self.vx * t, self.y + self.vy * t


class HighwayEnv:
    """
    3-lane straight highway with configurable obstacle vehicles.

    Parameters
    ----------
    dt       : simulation time step [s]
    scenario : 'cruise' | 'overtake' | 'dense'
    """

    def __init__(self, dt: float = 0.1, scenario: str = 'cruise'):
        self.dt       = dt
        self.scenario = scenario
        self.time     = 0.0
        self.obstacles: list[ObstacleVehicle] = []
        self._setup_scenario(scenario)

    def _setup_scenario(self, scenario: str):
        if scenario == 'cruise':
            # FIX: removed slow lead car from y=0 (ego lane).
            # Cruise = free-road speed tracking test.
            # One faster car in the adjacent lane (y=4) provides mild
            # lateral awareness without blocking the ego lane.
            self.obstacles = [
                ObstacleVehicle(x=40.0, y=4.0, vx=15.0),  # adjacent lane, faster
                ObstacleVehicle(x=80.0, y=4.0, vx=18.0),  # further ahead, fast
            ]
            self.ego_init = np.array([0.0, 0.0, 0.0, 15.0])
            self.v_ref    = 20.0
            self.lane_ref = 0

        elif scenario == 'overtake':
            # FIX: ego-lane cars spaced further apart so there is a
            # passable gap. y=4 blocker moved forward to x=80 so the
            # MPC can safely use y=4 during the early overtake window.
            self.obstacles = [
                ObstacleVehicle(x=20.0,  y=0.0, vx=7.0),   # close slow car, ego lane
                ObstacleVehicle(x=80.0,  y=0.0, vx=7.0),   # far slow car, ego lane
                ObstacleVehicle(x=80.0,  y=4.0, vx=18.0),  # fast car, adjacent lane (far)
                ObstacleVehicle(x=70.0,  y=8.0, vx=12.0),  # outer lane
            ]
            self.ego_init = np.array([0.0, 0.0, 0.0, 15.0])
            self.v_ref    = 22.0
            self.lane_ref = 0

        elif scenario == 'dense':
            # Unchanged — dense random traffic tests full avoidance capability
            np.random.seed(42)
            self.obstacles = []
            for lane_y in LANE_CENTERS:
                for xi in np.arange(10, 200, 25):
                    vx = np.random.uniform(8, 18)
                    self.obstacles.append(
                        ObstacleVehicle(x=xi, y=lane_y, vx=vx)
                    )
            self.ego_init = np.array([0.0, 0.0, 0.0, 12.0])
            self.v_ref    = 20.0
            self.lane_ref = 0

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def reset(self):
        """Reset environment and return initial ego state."""
        self._setup_scenario(self.scenario)
        self.time = 0.0
        return self.ego_init.copy()

    def step(self, ego_state: np.ndarray, ego_control: np.ndarray):
        """Apply ego control, advance all obstacle vehicles one step."""
        from vehicle_model import bicycle_dynamics
        new_ego = bicycle_dynamics(ego_state, ego_control, self.dt)
        for obs in self.obstacles:
            obs.step(self.dt)
        self.time += self.dt
        return new_ego

    def get_nearby_obstacles(self, ego_state: np.ndarray,
                              horizon: int = 0,
                              radius: float = 60.0) -> list:
        """Return obstacles within radius of ego, optionally time-shifted."""
        ex, ey = ego_state[0], ego_state[1]
        t = horizon * self.dt
        result = []
        for obs in self.obstacles:
            px, py = obs.predict(t)
            if abs(px - ex) < radius and abs(py - ey) < radius * 0.6:
                result.append((px, py, OBS_RADIUS))
        return result

    def get_lane_center(self, lane_idx: int = None) -> float:
        """Return y-coordinate of lane centre."""
        if lane_idx is None:
            lane_idx = self.lane_ref
        return LANE_CENTERS[lane_idx % N_LANES]

    def closest_lane(self, y: float) -> int:
        """Return index of lane closest to y."""
        dists = [abs(y - lc) for lc in LANE_CENTERS]
        return int(np.argmin(dists))

    def is_collision(self, ego_state: np.ndarray) -> bool:
        ex, ey = ego_state[0], ego_state[1]
        for obs in self.obstacles:
            if (abs(ex - obs.x) < OBS_LENGTH and
                    abs(ey - obs.y) < OBS_WIDTH):
                return True
        return False