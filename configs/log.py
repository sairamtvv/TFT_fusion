import logging
import logging.handlers
import os
import sys
import time


def setup_logging(config):
    log_config = config
    log_file_name = log_config.log_name_prefix + time.strftime("%Y%m%d-%H%M%S") + ".log"

    dir_path = os.getcwd()
    log_comp_directory = os.path.join(dir_path, log_config.log_directory)

    # Check if directory exist, else create it
    if not os.path.exists(log_comp_directory):
        os.mkdir(log_comp_directory)

    logfile_path = os.path.join(log_comp_directory, log_file_name)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName(log_config.log_level))
    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)-45s"
        " " + "-" + " " + "%(levelname)-8s — %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)

    file_handler = logging.FileHandler(logfile_path, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
