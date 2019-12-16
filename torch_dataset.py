import torch
from dataset import sentence_begin_tag, sentence_end_tag, \
                    unknown_word_tag, sentence_pad_tag

class DialogDataset(torch.utils.data.Dataset):

    def __init__(self, srcWordsList, tgtWordsList, vocab=None, 、
                 w2i=None, i2w=None, include_bos8eos=True, max_stc_len=None):
        if vocab is not None:
            self.i2w = vocab  # should include unk, pad, bos, eos and so on
            self.w2i = {}
            for i, k in enumerate(self.i2w):
                self.w2i[k] = i
        else:
            assert w2i is not None and i2w is not None
            self.w2i = w2i
            self.i2w = i2w
        
        self.bos_token = sentence_begin_tag
        self.eos_token = sentence_end_tag
        self.pad_token = sentence_pad_tag
        self.unk_token = unknown_word_tag
        self.srcWordsList = srcWordsList
        self.tgtWordsList = tgtWordsList
        self.include_bos8eos = include_bos8eos
        self.max_stc_len = max_stc_len
        
        self.srcList = self.wordsList2indicesList(srcWordsList)
        self.tgtList = self.wordsList2indicesList(tgtWordsList)
            
        assert len(self.srcList) == len(self.tgtList)
        
    def wordsList2indicesList(wordsList):
        indicesList = []
        for words in wordsList:
            if self.max_stc_len is not None:
                words = words[self.max_stc_len - 2] if self.include_bos8eos \
                   else words[self.max_stc_len]
            if self.include_bos8eos:
                words = [self.bos_token] + words + [self.eos_token]
            indices = [self.w2i.get(word, self.w2i[self.unk_token]) for e in words]
            indicesList.append(indices)
        return indicesList

    def __len__(self):
        return len(self.srcList)

    def __getitem__(self, idx):
        return self.srcList[idx], self.tgtList[idx]
    
    
def dialog_collate_fn(batch, pad_idx=0):
    qs, rs = zip(*batch)
    q_len = [len(q) for q in qs]
    r_len = [len(r) for r in rs]
    max_q_len = max(q_len)
    max_r_len = max(r_len)
    q_batch, r_batch = [], []
    for q, r in batch:
        q = q + [pad_idx] * (max_q_len - len(q))
        r = r + [pad_idx] * (max_r_len - len(r))
        q_batch.append(q)
        r_batch.append(r)
    q_tensor = torch.LongTensor(q_batch)
    r_tensor = torch.LongTensor(r_batch)
    q_len_tensor = torch.LongTensor(q_len)
    r_len_tensor = torch.LongTensor(r_len)
    q_tensor.transpose_(0, 1)
    r_tensor.transpose_(0, 1)
    return q_tensor, r_tensor, q_len_tensor, r_len_tensor