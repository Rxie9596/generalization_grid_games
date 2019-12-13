from generalization_grid_games.envs import maze_navigation_simple as mns
from generalization_grid_games.envs.utils import run_random_agent_demo



def run_interactive_demos():
    mns.MazeNavigation_simpleGymEnv0(interactive=True)
    mns.MazeNavigation_simpleGymEnv1(interactive=True)
    mns.MazeNavigation_simpleGymEnv2(interactive=True)
    mns.MazeNavigation_simpleGymEnv3(interactive=True)
    mns.MazeNavigation_simpleGymEnv4(interactive=True)
    mns.MazeNavigation_simpleGymEnv5(interactive=True)
    mns.MazeNavigation_simpleGymEnv6(interactive=True)
    mns.MazeNavigation_simpleGymEnv7(interactive=True)


if __name__ == "__main__":
    run_interactive_demos()
    # run_random_agent_demo(c.ChaseGymEnv1)
