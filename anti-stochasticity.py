import os
import random


def seed_everything(seed=1234):
    """ 受启发修改：https://www.kaggle.com/duykhanh99/lstm-fast-ai-tuning """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import np
        np.random.seed(seed)
    except:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    except:
        pass