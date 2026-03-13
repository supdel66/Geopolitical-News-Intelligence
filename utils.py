import time
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PipelineLogger")

def timer_logger(func):
    """
    A decorator that logs the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Starting execution of '{func.__name__}'")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Finished execution of '{func.__name__}' in {elapsed_time:.4f} seconds.")
            return result
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.error(f"Error executing '{func.__name__}' after {elapsed_time:.4f} seconds: {e}")
            raise
    return wrapper
