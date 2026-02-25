import gymnasium as gym


class FourToFiveTupleWrapper(gym.Wrapper):
    """Adapts envs that return 4-tuples to the gymnasium 5-tuple API.

    Our RingRoadEnv.step() returns (obs, reward, done, info).
    Gymnasium wrappers (ClipAction, NormalizeReward, etc.) expect
    (obs, reward, terminated, truncated, info).

    This wrapper bridges that gap without modifying the env itself,
    which would break Q-learning/DQN scripts that expect 4-tuples.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        # If reset returns just obs, wrap it
        return result, {}
