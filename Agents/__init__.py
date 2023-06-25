from .DQNAgent import DQNAgent
from .DQNRewardAgent import DQNRewardAgent
from .ACAgent import ACAgent

import sys

def get_agent(classname):
    return getattr(sys.modules[__name__], classname)
