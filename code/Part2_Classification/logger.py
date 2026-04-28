import logging

def get_logger(name, log_file=None):
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(levelname)s - %(message)s"
        )

        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)


        logger.propagate = False

    return logger
