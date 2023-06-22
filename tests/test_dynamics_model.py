
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import parser, dist_utils, misc
from utils.config import *
from utils.logger import *
import time
from tools import builder
import torch
from dynamics_dataset_v2 import FFN_dynamics_v2
import yaml_config_override
from easydict import EasyDict
from models.pointnet2_sem_seg import get_model
import torch.nn as nn
args = parser.get_args()
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
logger = get_root_logger(log_file=log_file, name=args.log_name)
config = get_config(args, logger = logger)
config.dataset.train.others.bs = config.total_bs
config.subset = "train"
config.DATA_PATH = "/data/sashank/datasets/easy_large_overfit/data"
config.PC_PATH = "/data/sashank/datasets/easy_large_overfit"
config.ver = "9D"
config.device = 'cuda:0'
args.distributed = False
(train_sampler, train_dataloader, train_dataset), (_, test_dataloader, test_dataset) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)

config.model.disable_batch_and_group_norm = config.total_bs < 32
config.model.disable_dropout_pointnet = True
pointr_model = builder.model_builder(config.model)
pointr_model.train()
def is_batchnorm_disabled(model):
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                return False
        return True
    
def is_dropout_disabled(model):
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            return False
    return True
print("Batchnorm in pointnet is disabled: ", is_batchnorm_disabled(pointr_model.pointnet))
print("Dropout in pointnet is disabled: ", is_dropout_disabled(pointr_model.pointnet))
for batch_idx, data in enumerate(train_dataloader):
    inp = data[3]['dyn_input'].to(torch.float32).to(config.device)
    inp = inp.permute(0,2,1)
    outp_pointr = pointr_model.pointnet(inp)
breakpoint()