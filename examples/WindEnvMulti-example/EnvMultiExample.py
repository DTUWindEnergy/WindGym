import numpy as np
from WindGym.wind_env_multi import WindFarmEnvMulti
from py_wake.examples.data.hornsrev1 import V80


def main():
    # Environment setup
    x_pos = [100, 500, 900]  # Turbine x positions
    y_pos = [0, 0, 0]  # Turbine y positions

    # Create the wind farm environment
    env = WindFarmEnvMulti(
        turbine=V80(),  # The turbine type (py_wake Turbine object)
        x_pos=x_pos,  # Turbine positions
        y_pos=y_pos,
        n_passthrough=10,  # Number of times wind flows over the farm per episode
        turbtype="Random",  # Turbulence type
        config="Env1_with_probes.yaml",  # Environment configuration file
        render_mode="human",  # Render mode ("human" or None)
    )

    print("Environment created successfully!")
    print(f"Agents: {env.possible_agents}")

    # Print action space info
    for agent in env.possible_agents:
        print(f"Agent {agent} action space: {env.action_space(agent)}")

    # Run simulation with random actions
    num_episodes = 3
    max_steps = 50

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")

        # Reset environment
        observations, info = env.reset()
        episode_rewards = {agent: 0.0 for agent in env.possible_agents}

        for step in range(max_steps):
            # Sample random actions for all agents
            actions = {}
            for agent in env.possible_agents:
                actions[agent] = env.action_space(agent).sample()

            # Take step in environment
            observations, rewards, dones, truncations, infos = env.step(actions)

            # Accumulate rewards
            for agent in env.possible_agents:
                episode_rewards[agent] += rewards[agent]

            # Print step information
            print(f"Step {step + 1}:")
            print(f"  Actions: {actions}")
            print(f"  Rewards: {rewards}")

            # Render environment
            env.render()

            # Check if episode is done
            if all(dones.values()) or all(truncations.values()):
                print(f"Episode ended at step {step + 1}")
                break

        # Print episode summary
        print(f"Episode {episode + 1} Summary:")
        for agent in env.possible_agents:
            print(f"  {agent}: {episode_rewards[agent]:.2f}")
        print(f"  Average reward: {np.mean(list(episode_rewards.values())):.2f}")

    print("\nSimulation completed!")


if __name__ == "__main__":
    main()
