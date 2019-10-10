import os
import random
import numpy as np
import torch
        

def seed_everything(seed=205, print=print):
    """ VARIFIED by Kaggle
    受启发修改：https://www.kaggle.com/duykhanh99/lstm-fast-ai-tuning
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'Everything seeded as seed = {seed}.')

    
def print_grad_params(model, print=print):
    """
    """
    for name, param in model.named_parameters():
        print(f'requires_grad={param.requires_grad}: {name}')

    
def visualize_model(model, print=print):
    """
    需要进一步拓展
    """
    print(model)
    print_grad_params(model, print=print)           


def has_nan(x: torch.Tensor, quit_if_nan=True, print=print):
    result = torch.isnan(x).detach().cpu().numpy().sum() == 0
    if quit_if_nan:
        assert not result, f'[ERROR] {x.name} has nan'
    else:
        if result:
            print(f'DANGER! {x.name} has nan')

    
def gpu2np(tensor):
    return tensor.detach().cpu().numpy()


def gpu2list(tensor):
    return gpu2np(tensor).tolist()