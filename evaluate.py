import torch, math, time, argparse, json, os, sys
import random, dataset, utils, losses, net
import numpy as np

from net.resnet import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from tqdm import *
import wandb

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

parser = argparse.ArgumentParser(description=
    'Official implementation of `IDML` on retrieval tasks'
)
parser.add_argument('--dataset', 
    default='cub', choices=('CT', 'cub', 'cars', 'SOP'),
    help = 'Evaluation dataset: CT, cub, cars, or SOP'
)
parser.add_argument('--embedding-size', default = 512, type = int,
    dest = 'sz_embedding',
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--batch-size', default = 150, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 4, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--model', default = 'resnet50', choices=('resnet18', 'resnet50', 'resnet101'),
    help = 'Model for training'
)
parser.add_argument('--l2-norm', default = 1, type = int,
    help = 'L2 normlization'
)
parser.add_argument('--resume', default = '',
    help = 'Path of resuming model'
)
parser.add_argument('--remark', default = '',
    help = 'Any reamrk'
)

parser.add_argument('--data-root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'), help='Parent of CT, CUB_200_2011, cars196, or Stanford_Online_Products.')
args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Data Root Directory
data_root = os.path.abspath(args.data_root)
    
# Dataset Loader and Sampler
ev_dataset = dataset.load(
        name = args.dataset,
        root = data_root,
        mode = 'eval',
        transform = dataset.utils.make_transform(
            is_train = False,
            is_inception = (args.model == 'bn_inception')
        ))

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size = args.sz_batch,
    shuffle = False,
    num_workers = args.nb_workers,
    pin_memory = True
)


# Backbone Model
model_class = {'resnet18': Resnet18, 'resnet50': Resnet50, 'resnet101': Resnet101}[args.model]
model = model_class(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)

if os.path.isfile(args.resume):
    print('=> loading checkpoint {}'.format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print('=> No checkpoint found at {}'.format(args.resume))
    sys.exit(0)
                    
with torch.no_grad():
    print("**Evaluating...**")
    if args.dataset != 'SOP':
        Recalls = utils.evaluate_cos(model, dl_ev)

    else:
        Recalls = utils.evaluate_cos_SOP(model, dl_ev)

    
