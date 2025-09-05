# from .cartpole import CartpoleEnv
# from .half_cheetah import HalfCheetahEnv
# from .pusher import PusherEnv
# from .reacher import Reacher3DEnv
import importlib

def get_item(name):
    dict = {
        "cartpole": "CartpoleEnv",
        "cartpole_stab": "CartpoleStabEnv",
        "cartdoublepole": "CartDoublePendulumEnv",
        "cartdoublepole_stab": "CartDoublePendulumStabEnv",
        "swimmer": "SwimmerEnv"
    }

    module = importlib.import_module("mymbrl.envs."+name)
    module_class = getattr(module, dict[name])
    return module_class