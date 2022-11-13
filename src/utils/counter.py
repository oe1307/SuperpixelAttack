import time
from functools import wraps

from .logging import setup_logger

logger = setup_logger(__name__)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        wrapper.process_time = time.time() - start
        return result

    return wrapper


def counter(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        wrapper.count += 1
        result = func(*args, **kargs)
        return result

    wrapper.count = 0
    return wrapper
