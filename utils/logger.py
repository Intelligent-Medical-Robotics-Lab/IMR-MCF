import logging
import datetime

def build_logger(logger_name, path, rank=None):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    if rank == 0 or rank is None:
        file_handler = logging.FileHandler(f'{path}/{current_time}.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
