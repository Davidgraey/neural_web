"""
description
https://packaging.python.org/en/latest/

"""
import logging

__version__ = "0.1.0"

# ________ set up logging ________
logger = logging.getLogger("neural_web")
logger.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                   level=logging.INFO)
f_handler = logging.FileHandler('neural_web.log')
f_handler.setLevel(logging.INFO)
logger.addHandler(f_handler)


# ________ set up public objects ________
__all__ = ["__version__"]
