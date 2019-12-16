import os
import random
import math

import numpy as np
import torch

from dataset import ids2words
from metrics.python_bleu import calculate_bleu
from metrics.Inter_Distinct import deal_by_all as distinct
        

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
        
        
def tensors2device(device, *args):
    return [arg.to(device) for arg in args]
        
        
def generate_one(model, q, r, q_len, r_len):
    model.eval()
    with torch.no_grad():
        tensors2device(q, r, q_len, r_len)
        output = model(q, r, q_len, r_len)
    return output

        
def generate_all(model, loader):
    outputs = []
    qs = []
    rs = []
    
    for q, r, q_len, r_len in tqdm(loader, desc='Generating'):
        output = generate_one(q, r, q_len, r_len)

        outputs.extend(gpu2list(output))
        qs.extend(gpu2list(q.transpose(0, 1)))
        rs.extend(gpu2list(r.transpose(0, 1)))

    return outputs, qs, rs


def seq2seq_evaluate_epoch(model, loader):
    total_loss = 0.
    with torch.no_grad():
        for q, r, q_len, r_len in tqdm(loader, desc='Evaluating'):
            total_loss += model(q, r, q_length, r_length).item()
    avg_loss = total_loss / len(loader)
    
    print(f'Evaluated. Average loss = {avg_loss}')
    return avg_loss


def seq2seq_train_epoch(model, optimizer, loader, writer, epoch, log_interval, clip_grad=None):
    '''
    clip_grad usually = 1.0
    '''
    model.train() # Turn on the train mode
    total_loss = 0.
    total_len = len(loader)
    start_time = time.time()
    
    for i, (q, r, q_len, r_len) in tqdm(loader):
        q, r, q_len, r_len = tensors2device(q, r, q_len, r_len)
        optimizer.zero_grad()
        loss = model(q, r, q_len, r_len)
        writer.add_scalar('Train_Loss', loss.detach(), (epoch - 1) * total_len + i)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        if i % log_interval == 0:
            inter_cases = 1 if i == 0 else log_interval
            cur_loss = total_loss / inter_cases
            elapsed = (time.time() - start_time) * 1000 / inter_cases
            print('epoch {:2d} | {:6d}/{:6d} batches | {:.3f}s/batch |'
                  ' lr {:5.5f} | loss {:5.3f} | ppl {:8.3f}'.format(
                    epoch, i, total_len, elapsed,
                    optimizer.param_groups[0]['lr'], cur_loss, math.exp(cur_loss)))
            total_loss = 0.
            start_time = time.time()


def seq2seq_train_all(model, optimizer, loader, writer, epochs, log_interval, \
                      i2w, ckpt_path: str, lr_decay=0.5, clip_grad=None):

    best_val_loss, best_epoch, best_bleu_epoch, best_bleu = float("inf"), 0, 0, 0.0
    keep_lr = True

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        '''train'''
        seq2seq_train_epoch(model, optimizer, loader, writer, epoch, \
                            log_interval, clip_grad)
        
        '''validation'''
        val_loss = seq2seq_evaluate_epoch(model, loader)
        writer.add_scalar('Val_Loss', val_loss.detach(), epoch)
        print('-' * 79)
        ppl = round(math.exp(val_loss), 2)
        print('Epoch {:2d} end | time: {:5.2f}s | '
              'dev loss/prev best {:5.2f}/{:5.2f}@{} | dev ppl {:4.2f}'.format(
                  epoch, (time.time() - epoch_start_time),
                  val_loss, best_val_loss, best_epoch, ppl))
        
        '''lr_decay'''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            keep_lr = True
        else:
            if not keep_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                keep_lr = True
            keep_lr = False
            
        '''save'''
        torch.save(model.state_dict(), f'{ckpt_path}/{epoch}-{ppl}')
        
        '''generate'''
        results, qs, rs = generate_all(model, loader)
        bleu_rs = []
        bleu_refers = []
        for r, q, refer in tqdm(zip(results, qs, rs)):
            bleu_r = id2word(r, ids2words(ids, i2w))
            bleu_refer = ids2words(refer, i2w)
            bleu_rs.append(bleu_r)
            bleu_refers.append([bleu_refer])
        bleu_all, bleu1, bleu2, bleu3, bleu4 = calculate_bleu(bleu_refers, bleu_rs)
        print(f'Bleu score = {bleu_all}, {bleu1} | {bleu2} | {bleu3} | {bleu4}')
        dist = distinct(bleu_rs)
        print(f'Distinct score = {dist}')

        writer.add_scalars(
            'Bleu', 
            {'all': bleu_all, '1': bleu1, '2': bleu2, '3': bleu3, '4': bleu4},
            epoch)
        writer.add_scalars('Distinction', dist, epoch)

        print(f'Bleu/prev best {bleu_all}/{best_bleu}@{best_bleu_epoch}')
        if bleu_all > best_bleu:
            best_bleu = bleu_all
            best_bleu_epoch = epoch

        with open(config.res_path / f'{ckpt_path}/{epoch}-{bleu_all}-{bleu1}',
                  'w', encoding='utf-8') as f:
            for r in str_rs:
                f.writelines(' '.join(r) + '\n')

        print(f'Epoch {epoch} end')
        print('-' * 79)