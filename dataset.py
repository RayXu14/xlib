from tqdm import tqdm
import random
        
        
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
            with open(path, encoding='utf-8') as f:
                for utt in tqdm(zip(fs, ft)):
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