'''
Custom errors for organelles / cells
'''
import Exception

import logging
logger = logging.getLogger("neural_web")


class DimensionError(ValueError):
    """Specific error for handling the case where array dimensions do not
    match as expected"""

