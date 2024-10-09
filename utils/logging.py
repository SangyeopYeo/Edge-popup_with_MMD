import logging


def LoggerSetting(opt):
    # Logging
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s(%(name)s): %(message)s")
    consH = logging.StreamHandler()
    consH.setFormatter(formatter)
    consH.setLevel(logging.DEBUG)
    logger.addHandler(consH)
    filehandler = logging.FileHandler(f"{opt.outf}_logfile.log")
    logger.addHandler(filehandler)
    request_file_handler = True
    log = logger
    return log
