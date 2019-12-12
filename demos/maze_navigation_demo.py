from generalization_grid_games.envs import maze_navigation as mn
from generalization_grid_games.envs.utils import run_random_agent_demo



def run_interactive_demos():
    mn.MazeNavigationGymEnv2(interactive=True)


if __name__ == "__main__":
    run_interactive_demos()
    # run_random_agent_demo(c.ChaseGymEnv1)
