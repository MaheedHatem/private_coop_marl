from .envs import CoinGatherEnv
from gym.envs.registration import register

register(
    id="CoinGathering-v0",
    entry_point="coin_gathering.coin_gathering.envs:CoinGatherEnv",
)