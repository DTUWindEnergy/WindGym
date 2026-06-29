import multiprocessing as mp

import numpy as np
from PIL import Image

from WindGym.wind_env_multi import WindFarmEnvMulti


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == "step":
                actions = data
                obs, rewards, dones, truncs, infos = env.step(actions)
                remote.send((obs, rewards, dones, truncs, infos))
            elif cmd == "render":
                env.render()
                remote.send(None)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "seed":
                seed = data
                env.seed(seed)
                remote.send(None)
            else:
                raise NotImplementedError(f"Unknown cmd {cmd}")
    except KeyboardInterrupt:
        pass


class EnvFnWrapper:
    def __init__(self, env_fn):
        self.env_fn = env_fn

    def __call__(self):
        return self.env_fn()


class ParallelPettingZooMultiprocessingWrapper:
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(
            *[mp.Pipe() for _ in range(self.num_envs)]
        )
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, EnvFnWrapper(env_fn))
            p = mp.Process(target=worker, args=args)
            p.daemon = True
            p.start()
            work_remote.close()
            self.processes.append(p)

        # Create dummy env to get agent metadata & spaces
        dummy_env = env_fns[0]()
        dummy_env.reset()

        self.possible_agents = list(dummy_env.possible_agents)
        self.agents = list(dummy_env.agents)  # current active agents

        self.observation_spaces = {
            agent: dummy_env.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: dummy_env.action_space(agent) for agent in self.possible_agents
        }
        dummy_env.close()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]

        obs_list, info_list = zip(*results)

        combined_obs = {
            agent: [obs.get(agent) for obs in obs_list]
            for agent in self.possible_agents
        }
        combined_info = {key: [info[key] for info in info_list] for key in info_list[0]}

        return combined_obs, combined_info

    def reset_done_envs(self, done_env_indices):
        """
        Reset only the environments with indices in done_env_indices.
        done_env_indices: list or set of ints - env indices that are done.
        Returns updated observations and infos with resets done in place.
        """
        # Send reset commands only to done env remotes
        for idx in done_env_indices:
            self.remotes[idx].send(("reset", None))

        # Collect reset results only for those envs
        reset_results = {idx: self.remotes[idx].recv() for idx in done_env_indices}

        # Now get current observations/info for all envs
        # For envs not reset, keep old obs/info (or store previous obs/info in your loop)
        # For envs reset, replace obs/info with reset results

        # Here you must have stored the previous obs/info outside of this method.
        # Let's assume you pass in the previous obs/info dicts or keep them as class variables.

        # For example, if you have:
        # self.current_obs, self.current_info

        for idx, (obs, info) in reset_results.items():
            for agent in self.possible_agents:
                self.current_obs[agent][idx] = obs.get(agent)
                for key in info:
                    self.current_info[key][idx] = info[key]

        return self.current_obs, self.current_info

    def step(self, actions):
        for i, remote in enumerate(self.remotes):
            env_actions = {
                agent: actions[agent][i]
                for agent in self.possible_agents
                if actions[agent][i] is not None
            }
            remote.send(("step", env_actions))

        results = [remote.recv() for remote in self.remotes]

        obs_list, rewards_list, dones_list, truncs_list, infos_list = zip(*results)

        combined_obs = {
            agent: [obs.get(agent) for obs in obs_list]
            for agent in self.possible_agents
        }
        combined_rewards = {
            agent: [rew.get(agent, 0.0) for rew in rewards_list]
            for agent in self.possible_agents
        }
        combined_dones = {
            agent: [done.get(agent, False) for done in dones_list]
            for agent in self.possible_agents
        }
        combined_truncs = {
            agent: [trunc.get(agent, False) for trunc in truncs_list]
            for agent in self.possible_agents
        }
        combined_infos = {
            agent: [info.get(agent, {}) for info in infos_list]
            for agent in self.possible_agents
        }

        return (
            combined_obs,
            combined_rewards,
            combined_dones,
            combined_truncs,
            combined_infos,
        )

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for p in self.processes:
            p.join()

    def seed(self, seed=None):
        for i, remote in enumerate(self.remotes):
            remote.send(("seed", None if seed is None else seed + i))
        for remote in self.remotes:
            remote.recv()

    def render(self):
        for remote in self.remotes:
            remote.send(("render", None))
        # Wait for all to finish rendering
        [remote.recv() for remote in self.remotes]

    def render_grid(self, mode="human", grid_shape=None):
        # Ask all envs for their current rendered frame
        for remote in self.remotes:
            remote.send(("render", "rgb_array"))
        frames = [remote.recv() for remote in self.remotes]

        # Convert to numpy arrays if needed
        frames = [np.array(f) for f in frames]

        # Auto-determine grid shape if not provided
        n = len(frames)
        if grid_shape is None:
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
        else:
            rows, cols = grid_shape

        # Resize all frames to the same size
        h, w = frames[0].shape[:2]
        resized_frames = [Image.fromarray(f).resize((w, h)) for f in frames]

        # Create blank canvas
        grid_img = Image.new("RGB", (cols * w, rows * h))

        # Paste each frame into the grid
        for idx, frame in enumerate(resized_frames):
            r = idx // cols
            c = idx % cols
            grid_img.paste(frame, (c * w, r * h))

        if mode == "human":
            grid_img.show()  # opens system image viewer
        elif mode == "rgb_array":
            return np.array(grid_img)
