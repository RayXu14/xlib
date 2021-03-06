import sys
import logging
import os
from pathlib import Path
from functools import partial
import warnings

from lxml import etree

open_utf8 = partial(open, encoding='utf-8')

class Redirect:
    '''
    受启发修改：https://i.loli.net/2017/09/06/59b00aa83fb78.png
    '''
    def __init__(self, target, path='default.log'):
        self.terminal = target
        self.log = open_utf8(path, 'a')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        pass
    

def redirect_std(path: str):
    print('Redirecting STDOUT and STDERR')
    sys.stdout = Redirect(sys.stdout, path + '.out')
    sys.stderr = Redirect(sys.stderr, path + '.err')
    

def check_output_dir(path: str):
    """ VERIFIED
    """
    if os.path.exists(path) and os.listdir(path):
        input(f"Output directory [{path}] already exists and is not empty. Will cover it. Continue? >")
        for p in os.listdir(path):
            try:
                print(f'removing {path}/{p}')
                os.remove(f'{path}/{p}')
                print('success')
            except:
                print('fail')
                pass
    if not os.path.exists(path):
        os.makedirs(path)


def check_output_dirs(paths):
    for path in paths:
        check_output_dir(path)
        
    
def set_logger(logname='xlib.log'):
    """ VERIFIED
    受启发修改：https://blog.csdn.net/u010895119/article/details/79470443
    """
    logger = logging.getLogger()
    logger.setLevel('INFO')

    BASIC_FORMAT = "[%(asctime)s|%(levelname)s] %(message)s"
    DATE_FORMAT = '%Y/%m/%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel('DEBUG')  # 也可以不设置，不设置就默认用logger的level

    fhlr = logging.FileHandler(logname, mode='w', encoding='utf-8') # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO')

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger


class Config(dict):
    """ VERIFIED by Kaggle
    同时支持用self[""]和self.访问
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
    def show(self, print=print):
        print(self)


def log_gpu_info(print=print):
    """
    https://zhuanfou.com/ask/77800004_063?answer_id=77800023_1052
    """
    print(os.popen('nvidia-smi').read())

    
def get_xml_root(path, recover=False):
    parser = etree.XMLParser(recover=recover)
    tree = etree.parse(path, parser)
    root = tree.getroot()
    return root
    # root.getchildren()可获得子节点列表
    # node.text可获得节点正文内容
    

def ignore_warnings():
    warnings.filterwarnings('ignore')