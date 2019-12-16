import random
import pickle as pkl
from collections import Counter
import sys

from tqdm import tqdm

from basic_io import open_utf8
        
        
def shuffle_2array_together(x, y, inplace=True):
    """ VERIFIED
    将两个数组以同样的顺序shuffle
    """
    combined = list(zip(x, y))
    random.shuffle(combined)
    if inplace:
        x[:], y[:] = zip(*combined)
    else:
        x, y = zip(*combined)
    return x, y


avg = lambda lst : sum(lst) / len(lst)


def calc_avg_tokens(*args):
    '''
    memory saving style
    '''
    lens = []
    for arg in args:
        if type(arg) = str:  # path
            with open_utf8(path) as f:
                for utt in tqdm(zip(fs, ft), desc='Calculating average tokens'):
                    words = utt.strip().split()
                    lens.append(len(words))
        elif type(arg) = list:
            assert len(arg) > 0, 'No content'
            if type(arg[0]) == list:  # tokens
                lens.extend([len(e) for e in arg])
            elif type(arg[0]) == str:  # raw utterance
                for e in arg:
                    lens.append(len(e.strip().split()))
            else:
                assert False, 'Unknown element type'
        else:
            assert False, 'Unknown type'
    print(f'avg_utt_len = {avg(lens)}')
    

def load_pkl(path):
    with open(path, 'rb') as f:
        content = pkl.load(f)
    print(f'Loaded from {path}')
        

def dump_pkl(path, content):
    with open(path, 'wb') as f:
        pkl.dump(content, f)
        

def uttList2wordsList(utts: list, method):
    wordsList = [' '.join(method(utt)) + '\n' for utt in utts]
    return wordsList
            
            
def load_nagisa():
    import nagisa


def nagisa_cut(utt: str):
    utt = utt.strip()
    words = nagisa.tagging(utt)
    return words.words
            
    
nlp = None
def load_spacy(model='en_core_news_sm'):
    '''
    https://spacy.io/models
    '''
    import spacy
    global nlp
    nlp = spacy.load("en_core_web_sm")


def spacy_cut(utt: str):
    utt = utt.strip()
    words = [w.text for w in nlp(utt.strip())]
    return words
        

def uttListFile2wordsListFile(origin_path: str, method, output_path=None):
    output_path = origin_path + '.tok' if output_path is None else output_path
    with open_utf8(origin_path) as f, open_utf8(output_path, 'w') as ftok:
        for line in tqdm(f, desc=f'Tokenizing {origin_path}'):
            words = method(line)
            ftok.write(' '.join(words) + '\n')


def overview_list(lst, name='list', n=3):
    print(f'{name} has {len(lst)} elements')
    if n > 0:
        print(f'Head {n} examples')
        print(lst[:n])
        print(f'Tail {n} examples')
        print(lst[-n:])
        

def init_simple_vocab(include_stctag=True):
    counter = Counter()
    if include_stctag:
        counter['<bos>'] = sys.maxsize
        counter['<eos>'] = sys.maxsize
    counter['<pad>'] = sys.maxsize
    counter['<unk>'] = sys.maxsize
    return counter


def load_wordsListFile(path, counter=None):
    if counter is not None:
        origin_cnt_size = len(counter)
    
    wordsList = []
    with open_utf8(path) as f:
        for line in tqdm(f, desc=f'Loading {path}'):
            words = line.strip().split()
            if counter is not None:
                counter.update(words)
            wordsList.append(words)
            
    overview_list(wordsList, name='Loaded set')
    
    if counter is None:
        return wordsList
    else:
        print(f'Word Counter size: {len(counter)}')
        print('10 most common words:')
        print(counter.most_common(10))
        return wordsList, counter
    

def counter2vocab(counter, max_vocab=None):
    max_vocab = len(counter) if max_vocab is None else max_vocab
    print(f'Final Vocab size: {max_vocab}/{len(counter)}')

    vocab = []
    for word, cnt in counter.most_common(max_vocab):
        vocab.append(word)
    return vocab
    
    
    