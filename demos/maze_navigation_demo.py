from generalization_grid_games.envs import maze_navigation as mn
from generalization_grid_games.envs.utils import run_random_agent_demo



def run_interactive_demos():
    mn.MazeNavigationGymEnv0(interactive=True)
    mn.MazeNavigationGymEnv1(interactive=True)
    mn.MazeNavigationGymEnv2(interactive=True)
    mn.MazeNavigationGymEnv3(interactive=True)
    mn.MazeNavigationGymEnv4(interactive=True)
    mn.MazeNavigationGymEnv5(interactive=True)
    mn.MazeNavigationGymEnv6(interactive=True)
    mn.MazeNavigationGymEnv7(interactive=True)


if __name__ == "__main__":
    run_interactive_demos()
    # run_random_agent_demo(c.ChaseGymEnv1)
