""" helper function

author axiumao
"""

import os
import random
import sys
import torch

import numpy as np

from torch.utils.data import DataLoader

from dataset import My_Dataset

def get_network(args, samples_per_cls=None):
    """ return given network """
    device = args.device if hasattr(args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()

    elif args.net == 'MambaWithDWT':
        from models.MambaWithDWT import MambaWithDWT, ModelConfigs
        cfg = ModelConfigs()
        net = MambaWithDWT(cfg).to("cuda")

    elif args.net == 'MambaWithDWT2':
        from models.MambaWithDWT2 import MambaWithDWT, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    
    
    elif args.net == 'MambaWithDWT3':
        from models.MambaWithDWT3 import MambaWithDWT, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    
    elif args.net == 'MambaWithDWT4':
        from models.MambaWithDWT4 import MambaWithDWT, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    
    elif args.net == 'MambaWithDWT5':
        from models.MambaWithDWT5 import MambaWithDWT, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithDWT(cfg).to("cuda")

    elif args.net == 'MambaWithDWT6':
        from models.MambaWithDWT6 import MambaWithDWT, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    
    elif args.net == 'MambaWithDWT7':
        from models.MambaWithDWT7 import MambaWithDWT, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    
    elif args.net == 'MambaWithDWT8':
        from models.MambaWithDWT8 import MambaWithDWT, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    
    elif args.net == 'MambaWithDWT9':
        from models.MambaWithDWT9 import MambaWithDWT, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    
    elif args.net == 'MambaWithDWT10':
        from models.MambaWithDWT10 import MambaWithDWT, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    
    elif args.net == 'MambaWithDWT11':
        from models.MambaWithDWT11 import MambaWithDWT, ModelConfigs
        cfg = ModelConfigs()
        net = MambaWithDWT(cfg).to("cuda")

    elif args.net == 'MambaWithDWT11a':
        from models.MambaWithDWT11a import MambaWithDWT, ModelConfigs
        cfg = ModelConfigs()
        net = MambaWithDWT(cfg).to("cuda")

    elif args.net == 'MambaWithDWT11b':
        from models.MambaWithDWT11b import MambaWithDWT, ModelConfigs
        cfg = ModelConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    elif args.net == 'MambaWithDWT11c':
        from models.MambaWithDWT11c import MambaWithDWT, ModelConfigs
        cfg = ModelConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    elif args.net == 'MambaWithDWT11d':
        from models.MambaWithDWT11d import MambaWithDWT, ModelConfigs
        cfg = ModelConfigs()
        net = MambaWithDWT(cfg).to("cuda")
    
    
    elif args.net == 'dwt_classifier':
        from models.dwt_classifier import SimpleDWTClassifier, MambaConfigs
        model_configs = MambaConfigs()
        net = SimpleDWTClassifier(config=model_configs)
    
    elif args.net == 'PatchingClassifier':
        from models.pactch_classifier import PatchingClassifier, ModelConfigs
        cfg = ModelConfigs()
        net = PatchingClassifier(cfg)
    
    elif args.net == 'MambaWithPatch':
        from models.MambaWithPatch import MambaWithPatch, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithPatch(cfg) 

    elif args.net == 'MambaWithPatch1':
        from models.MambaWithPatch1 import MambaWithPatch, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithPatch(cfg)
    
    elif args.net == 'MambaWithPatch2':
        from models.MambaWithPatch2 import MambaWithPatch, MambaConfigs
        cfg = MambaConfigs()
        net = MambaWithPatch(cfg)

    elif args.net == 'MambaWithPatch3':
        from models.MambaWithPatch3 import MambaWithPatch, ModelConfigs
        cfg = ModelConfigs()
        net = MambaWithPatch(cfg)

    elif args.net == 'onlymamba':
        from models.onlymamba import MambaWithDWT, ModelConfigs
        cfg = ModelConfigs()
        net = MambaWithDWT(cfg)

    elif args.net == 'TransWithPatch':
        from models.TransWithPatch import ModelConfigs , TransformerWithPatch
        cfg = ModelConfigs()
        net = TransformerWithPatch(cfg)




    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net.to(device)  # 统一在此处设置设备


def get_mydataloader(pathway, data_id = 1, batch_size=16, num_workers=2, shuffle=True):
    Mydataset = My_Dataset(pathway, data_id, transform=None)
    Data_loader = DataLoader(Mydataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size) # DataLoader 是 PyTorch 提供的一个数据加载器，用于将数据集中的数据转换为可用于训练的批次。
    
    return Data_loader


def get_weighted_mydataloader(pathway, data_id = 1, batch_size=16, num_workers=2, shuffle=True):
    """
    获取带权重的数据加载器，用于处理不平衡数据集。
    （已修正类别计数逻辑）
    """
    Mydataset = My_Dataset(pathway, data_id, transform=None)
    
    # 这行代码的效率较低，因为它会完整遍历一次数据集来获取所有标签。
    # 如果您的 Mydataset 对象内部直接存储了标签列表（例如 self.labels），
    # 直接使用 Mydataset.labels 会快很多。但当前写法也能正常工作。
    all_labels = [label for data, label in Mydataset]
    
    # ======================== 核心修正部分 开始 ========================
    
    # 1. 定义好您项目的总类别数，这是一个非常重要的参数
    num_classes = 5
    
    # 2. 使用 np.bincount 替换 np.unique
    #    np.bincount 会统计从0到最大标签值每个整数出现的次数。
    #    通过设置 minlength=num_classes，我们强制它返回一个长度为5的数组。
    #    即使 all_labels 中缺少某个类别（比如没有4），返回数组的第4个位置也会是0，
    #    从而保证了 number 数组的长度始终为5。
    number = np.bincount(all_labels, minlength=num_classes)
    
    # ======================== 核心修正部分 结束 ========================

    # 计算权重。这里增加了一个极小值 1e-8，以防止当某个类别的样本数 number[i] 为0时，
    # 出现“除以零”的错误，这会让代码更健壮。
    weight = 1. / (torch.from_numpy(number).float() + 1e-8) 
    
    # 使用softmax归一化权重
    weight = torch.softmax(weight, dim=0)
    
    Data_loader = DataLoader(Mydataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return Data_loader, weight, number


def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)
