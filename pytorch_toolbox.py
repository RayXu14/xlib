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


class Regularization(torch.nn.Module):
    """
    https://blog.csdn.net/guyuealian/article/details/88426648
    """
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
 
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
 
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")
