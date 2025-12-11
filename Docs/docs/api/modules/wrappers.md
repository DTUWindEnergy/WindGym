# WindGym.wrappers Module

This module contains Gymnasium wrappers for WindGym environments.

<a id="module-WindGym.wrappers"></a>

## RecordEpisodeVals

### *class* WindGym.wrappers.record_episode_vals.RecordEpisodeVals(env: [VectorEnv](https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv), buffer_length=100)

Bases: [`RecordEpisodeStatistics`](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.RecordEpisodeStatistics)

This wraps the RecordEpisodeStatistics Wrapper.
It also adds a queue to store the mean power of the episodes. This is used for the logging during training.
Could also be expanded upon to include more statistics if wanted.

#### \_\_init_\_(env: [VectorEnv](https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv), buffer_length=100)

This wrapper will keep track of cumulative rewards and episode lengths.

* **Parameters:**
  * **env** (*Env*) – The environment to apply the wrapper
  * **buffer_length** – The size of the buffers `return_queue`, `length_queue` and `time_queue`
  * **stats_key** – The info key to save the data

#### reset(seed: [int](https://docs.python.org/3/library/functions.html#int) | [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, options: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Resets the environment using kwargs and resets the episode returns and lengths.

#### step(actions: ActType) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ObsType, ArrayType, ArrayType, ArrayType, [dict](https://docs.python.org/3/library/stdtypes.html#dict)]

Steps through the environment, recording the episode statistics.

## CurriculumWrapper

### *class* WindGym.wrappers.curriculum_wrapper.CurriculumWrapper(env: ~gymnasium.core.Env, n_envs: int, similarity_type: str = 'normalized_l2', yaw_check: str = 'current', weight_function=<function CurriculumWrapper.<lambda>>, huber_kappa: float = 1.0, exp_alpha: float = 1.0)

Bases: [`Wrapper`](https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper)

Curriculum wrapper for the WindGym environment.
This wrapper adds a curriculum-based similarity reward between the agent’s yaw
vector and a reference (“good”) yaw vector produced by a PyWakeAgent.
yaw_check options:

> - ‘current’: use the current yaw angles of the agent
> - ‘goal’: use the yaw angles that would have been used, with no yaw step limits (only for wind actions)

similarity_type options:
: - ‘l2’: negative L2 distance
  - ‘l1’: negative mean absolute error
  - ‘mse’: negative mean squared error
  - ‘normalized_l2’: 1 - (L2 distance / max_distance)
  - ‘exponential’: exp(-alpha \* L2 distance)
  - ‘cosine’: cosine similarity
  - ‘huber’: negative Huber loss

weight_function:
: function(step: int) -> float in [0,1], weighting env reward vs. similarity
  1 = env reward, 0 = similarity

#### \_\_init_\_(env: ~gymnasium.core.Env, n_envs: int, similarity_type: str = 'normalized_l2', yaw_check: str = 'current', weight_function=<function CurriculumWrapper.<lambda>>, huber_kappa: float = 1.0, exp_alpha: float = 1.0)

Wraps an environment to allow a modular transformation of the [`step()`](#WindGym.wrappers.curriculum_wrapper.CurriculumWrapper.step) and [`reset()`](#WindGym.wrappers.curriculum_wrapper.CurriculumWrapper.reset) methods.

* **Parameters:**
  **env** – The environment to wrap

#### reset(\*\*kwargs)

Reset the environment and the pywake agent.

#### step(action)

Take a step in the environment and calculate the reward based on the similarity between the yaw angles of the agent and the pywake agent.

## PowerWrapper

### *class* WindGym.wrappers.power_wrapper.PowerWrapper(env: ~gymnasium.core.Env, n_envs: int, weight_function=<function PowerWrapper.<lambda>>)

Bases: [`Wrapper`](https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper)

PowerWrapper wrapper for the WindGym environment.
This wrapper adds a reward based on the power from PyWake.

weight_function:
: function(step: int) -> float in [0,1], weighting env reward vs. similarity
  1 = env reward, 0 = similarity

#### \_\_init_\_(env: ~gymnasium.core.Env, n_envs: int, weight_function=<function PowerWrapper.<lambda>>)

Wraps an environment to allow a modular transformation of the [`step()`](#WindGym.wrappers.power_wrapper.PowerWrapper.step) and [`reset()`](#WindGym.wrappers.power_wrapper.PowerWrapper.reset) methods.

* **Parameters:**
  **env** – The environment to wrap

#### reset(\*\*kwargs)

Reset the environment and the pywake agent.

#### step(action)

Take a step in the environment and calculate the reward based on the power from PyWake, and the normal WindGym reward.

## AdversaryWrapper

### *class* WindGym.wrappers.adversary_wrapper.AdversaryWrapper(env: [VectorEnv](https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv), buffer_length=100)

Bases: [`RecordEpisodeStatistics`](https://gymnasium.farama.org/api/vector/wrappers/#gymnasium.wrappers.vector.RecordEpisodeStatistics)

This wraps the RecordEpisodeStatistics Wrapper.
It also adds a queue to store the mean power of the episodes. This is used for the logging during training.
Could also be expanded upon to include more statistics if wanted.

#### \_\_init_\_(env: [VectorEnv](https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv), buffer_length=100)

This wrapper will keep track of cumulative rewards and episode lengths.

* **Parameters:**
  * **env** (*Env*) – The environment to apply the wrapper
  * **buffer_length** – The size of the buffers `return_queue`, `length_queue` and `time_queue`
  * **stats_key** – The info key to save the data

#### reset(seed: [int](https://docs.python.org/3/library/functions.html#int) | [list](https://docs.python.org/3/library/stdtypes.html#list)[[int](https://docs.python.org/3/library/functions.html#int)] | [None](https://docs.python.org/3/library/constants.html#None) = None, options: [dict](https://docs.python.org/3/library/stdtypes.html#dict) | [None](https://docs.python.org/3/library/constants.html#None) = None)

Resets the environment using kwargs and resets the episode returns and lengths.

#### step(actions: ActType) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[ObsType, ArrayType, ArrayType, ArrayType, [List](https://docs.python.org/3/library/typing.html#typing.List)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]]

Steps through the environment, recording the episode statistics.
