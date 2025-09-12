# adagram_optimizers/__init__.py

# Import the utils package as a submodule
from . import utils

# Import important classes or functions from submodules for easier access
from .AdagramBase import *
from .AdagramPS import *
from .AdagramSVD import *
from .AdagramVanilla import *
from .DiagAdagrad import *
from .FullAdagrad import *
from .Kate import *
from .libcontext import *
from .Shampoo import *



# Optionally, define __all__ to specify what 'from adagram_optimizers import *' imports
# You should list the actual class names from your modules here.
# For example, if AdagramPS.py contains a class named 'AdagramPS', you would include that.
__all__ = [
    'AdagramPS', 
    'AdagramSVD',
    'AdagramVanilla',
    'DiagAdagrad',
    'FullAdagrad',
    'Kate',
    'Shampoo',
    'utils'
    # Add other class names as needed
]