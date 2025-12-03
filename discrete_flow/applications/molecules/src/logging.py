# Import public modules
import os
import logging

def define_logger(log_file_path:str, 
                  file_logging_level:str='INFO', 
                  stream_logging_level:str='DEBUG') -> object:
    """
    Define a logger object and return it.
    
    Args:
        log_file_path (str or Path): Path in which the log file should be stored in.
        file_logging_level (str): Logging level for the logs that are stored to the logfile (file).
        stream_logging_level (str): Logging level for the logs that are displayed (stream).

    Return:
        (logging.logger): Logger object.
    
    """
    # Check that the logging levels are expected
    expected_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if file_logging_level not in expected_levels:
        err_msg = f"The passed (uppercase) file logging level '{file_logging_level.upper()}' is not one of the expected logging level: {expected_levels}"
        raise ValueError(err_msg)
    if stream_logging_level not in expected_levels:
        err_msg = f"The passed (upercase) stream logging level '{stream_logging_level.upper()}' is not one of the expected logging level: {expected_levels}"
        raise ValueError(err_msg)

    # Remove the logfile if there already exists one
    if os.path.isfile(log_file_path):
        os.remove(log_file_path)

    # Turn the file and stream logging levels from strings to actual logging level objects
    file_logging_level   = getattr(logging, file_logging_level.upper())
    stream_logging_level = getattr(logging, stream_logging_level.upper())

    # Get the log file name and use it as name for the logger
    log_file_name = os.path.split(log_file_path)[1]
    logger_name   = log_file_name.removesuffix('.log')

    # Set the root logger's logging level to DEBUG
    # Remark: 1) For each logging event, the root logger's logging level (global) is used to determine if the
    #            event should be logged or not. 
    #         2) Thus, this 'global' logging level oversteers in some sence the 'local' logging levels of the handlers defined below.
    #            As the handler levels are explicitly set below, this should not happen so use the lowest level (DEBUG),
    logging.basicConfig(level=logging.DEBUG)

    # Initialize the logger
    logger = logging.getLogger(logger_name)

    # Remove handlers if the logger has any
    logger.handlers = []

    # Generate a file handler that will store logging information to the log file
    # and add it to the logger
    f_handler = logging.FileHandler(log_file_path)
    f_format  = logging.Formatter(fmt='[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    f_handler.setFormatter(f_format)
    f_handler.setLevel(file_logging_level)
    logger.addHandler(f_handler)

    # Generate a stream handler that will show the logging info to the user
    # and add it to the logger
    s_handler = logging.StreamHandler()
    s_format  = logging.Formatter(fmt='[%(levelname)s]: %(message)s')
    s_handler.setFormatter(s_format)
    s_handler.setLevel(stream_logging_level)
    logger.addHandler(s_handler)

    # Do not propagate logs from the file handlers to the base logger 
    # (so that only the handlers log but not the base logger)
    logger.propagate = False

    return logger