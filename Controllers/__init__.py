from .BaseController import BaseController
from .DecentralizedController import DecentralizedController
from .CentralizedController import CentralizedController

import sys

def get_controller(classname):
    return getattr(sys.modules[__name__], classname)