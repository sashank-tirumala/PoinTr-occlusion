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
args.distributed = False
(train_sampler, train_dataloader, train_dataset), (_, test_dataloader, test_dataset) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
dyn_dataset = FFN_dynamics_v2(config)
dyn_dataloader = torch.utils.data.DataLoader(dyn_dataset, batch_size=1, shuffle=True, num_workers=0)
a11, a12,b11, b12 = 0,0,0,0
for batch_idx, data in enumerate(train_dataloader):
    a11 = data[3]['dyn_input']
    a12 = data[3]['dyn_output']

for batch_idx, data in enumerate(dyn_dataloader):
    b11 = data[2]['dyn_input']
    b12 = data[2]['dyn_output']

assert torch.allclose(a11, b11), "Dataloader Dyn Input not working"
assert torch.allclose(a12, b12), "Dataloader Dyn Output not working"
print("Dataloader working")