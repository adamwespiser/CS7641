import gym
from gym.envs.registration import register

from .cliff_walking import *
from .frozen_lake import *

__all__ = ['RewardingFrozenLakeEnv', 'WindyCliffWalkingEnv']

register(
    id='RewardingFrozenLake-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '4x4'},
)

register(
    id='RewardingFrozenLake8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8'}
)
register(
    id='RewardingFrozenLake20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20'}
)

register(
    id='RewardingFrozenLakeNoRewards20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'rewarding': False}
)

register(
    id='RewardingFrozenLakeNoRewards8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'rewarding': False}
)

register(
    id='WindyCliffWalking-v0',
    entry_point='environments:WindyCliffWalkingEnv',
)

## FLake
register(
    id='FLake8x8-v0',
    entry_point='environments:FLakeEnv',
    kwargs={'map_name': '8x8'}
)
register(
    id='FLake20x20-v0',
    entry_point='environments:FLakeEnv',
    kwargs={'map_name': '20x20'}
)

def get_flake_env():
    return gym.make('FLake8x8-v0')

def get_large_flake_env():
    return gym.make('FLake20x20-v0')


## FLake ## V3
register(
    id='FLake8x8-v3',
    entry_point='environments:FLakeEnv',
    kwargs={'map_name': '8x8',
            'step_reward' : -0.01,
            'hole_reward' : -0.01,
            'start_reward' : -0.01,
            'goal_reward' : 1
            }
)
def get_flake_env_v3():
    return gym.make('FLake8x8-v3')

register(
    id='FLake20x20-v3',
    entry_point='environments:FLakeEnv',
    kwargs={'map_name': '20x20',
            'step_reward' : -0.01,
            'hole_reward' : -0.01,
            'start_reward' : -0.01,
            'goal_reward' : 5
            }
)
def get_large_flake_env_v3():
    return gym.make('FLake20x20-v3')

register(
    id='FLake20x20-v4',
    entry_point='environments:FLakeEnv',
    kwargs={'map_name': '20x20b',
            'step_reward' : -0.01,
            'hole_reward' : -0.1,
            'start_reward' : -0.01,
            'goal_reward' : 10
            }
)
def get_large_flake_env_v4():
    return gym.make('FLake20x20-v4')



# here
def get_rewarding_frozen_lake_environment():
    return gym.make('RewardingFrozenLake8x8-v0')

def get_large_rewarding_frozen_lake_environment():
    return gym.make('RewardingFrozenLake20x20-v0')


def get_frozen_lake_environment():
    return gym.make('FrozenLake-v0')


def get_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards8x8-v0')


def get_large_rewarding_no_reward_frozen_lake_environment():
    return gym.make('RewardingFrozenLakeNoRewards20x20-v0')


def get_cliff_walking_environment():
    return gym.make('CliffWalking-v0')


def get_windy_cliff_walking_environment():
    return gym.make('WindyCliffWalking-v0')

def get_rewarding_frozen_lake_8x8_environment():
    return gym.make('RewardingFrozenLake8x8')


