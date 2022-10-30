import time
from functools import wraps

from .logging import setup_logger

logger = setup_logger(__name__)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        process_time = time.time() - start
        print(f"{func.__name__} : {process_time} (sec)")
        return result

    return wrapper
