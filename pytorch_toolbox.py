import os
import random
import numpy as np
import torch
        

def seed_everything(seed=303, print=print):
    """
    受启发修改：https://www.kaggle.com/duykhanh99/lstm-fast-ai-tuning
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('Everything seeded.')
    
def print_grad_params(model, print=print):
    """
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    
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

            
def count_params(model, sample_input, print=print):
    """
    统计模型的参数量和计算量
    """
    from thop import profile
    flops, params = profile(model, input=(sample_input, ))
    print(f'FLOPS = {flops} PARAMS = {params}')
