import os
import random
import numpy as np
import torch
        

def seed_everything(seed=1234):
    """ 受启发修改：https://www.kaggle.com/duykhanh99/lstm-fast-ai-tuning """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True