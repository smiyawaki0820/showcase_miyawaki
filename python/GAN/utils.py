import random
import numpy as np

import torch
from torch.autograd import Variable # 自動微分用

def to_cuda(target):
    return target.cuda() if torch.cuda.is_available() else target

def to_variable(target, volatile=False):
    return Variable(to_cuda(target), volatile=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

