import pygame
import numpy as np

from highway_env import HighwayEnv, LANE_CENTERS
from mpc_controller import MPCController
from batch_planner import BatchPlanner

WIDTH, HEIGHT = 1000, 600
FPS = 60
DT = 0.1

SCALE = 5.0
OFFSET_X = 50
OFFSET_Y = 300


def world_to_screen(x, y):
    return int(x * SCALE + OFFSET_X), int(OFFSET_Y - y * SCALE)


def draw_lane_lines(screen):
    for y in LANE_CENTERS:
        _, sy = world_to_screen(0, y)
        pygame.draw.line(screen, (200, 200, 200), (0, sy), (WIDTH, sy), 2)


def draw_car(screen, x, y, color):
    sx, sy = world_to_screen(x, y)
    pygame.draw.rect(screen, color, (sx - 10, sy - 5, 20, 10))


def draw_traj(screen, traj, color, width=1):
    for i in range(len(traj) - 1):
        x1, y1 = world_to_screen(traj[i][0], traj[i][1])
        x2, y2 = world_to_screen(traj[i+1][0], traj[i+1][1])
        pygame.draw.line(screen, color, (x1, y1), (x2, y2), width)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    env = HighwayEnv(dt=DT, scenario="overtake")
    mpc = MPCController(N=15, dt=DT)
    batch = BatchPlanner(mpc)

    state = env.reset()
    v_ref = env.v_ref

    last_control = np.array([0.0, 0.0])

    running = True
    while running:
        screen.fill((30, 30, 30))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        best_res, all_res = batch.plan(state, v_ref, env.obstacles)

        if best_res is not None:
            u = best_res['u_apply']
            last_control = u
        else:
            u = last_control
            u[0] = max(u[0], 0.1)
            u[1] = 0.0

        state = env.step(state, u)

        # stability fixes
        state[3] = max(state[3], 5.0)
        state[1] = np.clip(state[1], -0.5, 9.5)

        draw_lane_lines(screen)

        if best_res is not None:
            for res in all_res:
                draw_traj(screen, res['trajectory'], (255, 0, 0), 1)
            draw_traj(screen, best_res['trajectory'], (0, 255, 0), 3)

        draw_car(screen, state[0], state[1], (0, 255, 255))

        for obs in env.obstacles:
            draw_car(screen, obs.x, obs.y, (255, 255, 0))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()