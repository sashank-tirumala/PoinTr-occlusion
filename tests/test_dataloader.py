import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import parser, dist_utils, misc
from utils.config import *
from utils.logger import *
import time
from tools import builder
import torch
args = parser.get_args()
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
logger = get_root_logger(log_file=log_file, name=args.log_name)
config = get_config(args, logger = logger)
config.dataset.train.others.bs = config.total_bs
args.distributed = False
(train_sampler, train_dataloader, train_dataset), (_, test_dataloader, test_dataset) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
for batch_idx, data in enumerate(train_dataloader):
    print(f"Batch {batch_idx}: {data}")
breakpoint()