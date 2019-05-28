import logging
import os


def check_output_dir(path, override=False):
    if os.path.exists(path) and os.listdir(path) and not override:
        input("Output directory () already exists and is not empty. Will cover it. Continue? >")
    if not os.path.exists(path):
        os.makedirs(path)

    
def set_logger(logname='xlib.log'):
    """ 受启发修改：https://blog.csdn.net/u010895119/article/details/79470443 """
    logger = logging.getLogger()
    logger.setLevel('INFO')

    BASIC_FORMAT = "[%(asctime)s|%(levelname)s] %(message)s"
    DATE_FORMAT = '%Y/%m/%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel('DEBUG')  # 也可以不设置，不设置就默认用logger的level

    fhlr = logging.FileHandler(logname, mode='w') # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO')

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger