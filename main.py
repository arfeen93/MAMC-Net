import argparse
import os
import torch
import json
import torch.backends.cudnn as cudnn
from torch import distributed as distributed
from data.mat_dataset import MatDataset
from data.domain_dataset import PACSDataset, DomainDataset
from methods import CuMix
from utils import test
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
from numpy import save
import paramiko
import random

ZSL_DATASETS = ['CUB', 'FLO', 'SUN', 'AWA1']
PACS_DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
DNET_DOMAINS = ['real', 'quickdraw', 'sketch', 'painting', 'infograph', 'clipart']

parser = argparse.ArgumentParser(description='Zero-shot Learning meets Domain Generalization -- ZSL experiments')
parser.add_argument('--dataset_name', default='domainnet', help='Dataset to run experiment on')
parser.add_argument('--target', default='cub', help='Which experiment to run (e.g. [cub, awa1, flo, sun, all])')
parser.add_argument('--zsl', action='store_true', help='ZSL setting?')
parser.add_argument('--dg', action='store_true', help='DG setting?')
parser.add_argument('--data_root', default='/mnt/c61a35bf-fc59-4aab-a996-b254f9ab9052/arfeen', type=str, help='Data root directory')
parser.add_argument('--name', default='test', type=str, help='Name of the experiment (used to store '
                                                           'the logger and the checkpoints)')
parser.add_argument('--runs', default=5, type=int, help='Number of runs per experiment')
parser.add_argument('--log_dir', default='./logs', type=str, help='Log directory')
parser.add_argument('--ckpt_dir', default='./checkpoints', type=str, help='Checkpoint directory')
parser.add_argument('--config_file', default=None, help='Config file for the method.')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--world_size", type = int, default=1, help='no of gpus to work on')

args = parser.parse_args()

"""multi-gpu params"""
# distributed.init_process_group(backend='nccl', init_method='env://')
#
# #world_size = torch.cuda.device_count()
# world_size = distributed.get_world_size()
# device_id, device = args.local_rank, torch.device(args.local_rank)
# rank  = distributed.get_rank()
# torch.cuda.set_device(device_id)
# world_info = {'world_size':world_size, 'rank':rank}
#import pdb;pdb.set_trace()
"""single gpu params"""
device_id, device = 0, torch.device("cuda:1")
world_size = 1 #args.world_size
rank = 0
world_info = {'world_size':world_size, 'rank':rank}

# Check association dataset--setting are correct + init remaining stuffs from configs
"Reproducibility"
np.random.seed(5)
random.seed(5)
#torch.random.seed(5)
torch.random.manual_seed(5)

assert args.dg or args.zsl, "Please specify if you want to benchmark ZSL and/or DG performances"

config_file = args.config_file
with open(config_file) as json_file:
    configs = json.load(json_file)
print(args.config_file)
multi_domain = False
input_dim = 2048
configs['freeze_bn'] = False

# Semantic W is used to rescale the principal semantic loss.
# Needed to have the same baseline results as https://github.com/HAHA-DL/Episodic-DG/tree/master/PACS for DG only exps
semantic_w = 1.0


if args.dg:
    target = args.target
    multi_domain = True
    if args.zsl:
        assert args.target in DNET_DOMAINS, \
            args.target + " is not a valid target domain in  DomainNet. Please specify a valid DomainNet target in " + DNET_DOMAINS.__str__()
        DOMAINS = DNET_DOMAINS
        dataset = DomainDataset
    else:
        assert args.target in PACS_DOMAINS, \
            args.target + " is not a valid target domain in PACS. Please specify a valid PACS target in " + PACS_DOMAINS.__str__()
        DOMAINS = PACS_DOMAINS
        dataset = PACSDataset
        input_dim = 512
        semantic_w = 3.0
        configs['freeze_bn']=True

    dataset_name = args.dataset_name
    if dataset_name == 'domainnet':
        sources = DOMAINS + []
        ns = len(sources)
        sources.remove(target)
        assert len(sources) < ns, 'Something is wrong, no source domains reduction after remove with target.'
    else:
        sources = ["real", "painting"]
        ns = len(sources)
        assert len(sources) == ns, 'Something is wrong, source domains reduction after remove with target.'
else:
    target = args.target.upper()
    assert target in ZSL_DATASETS, \
        args.target + " is not a valid ZSL dataset. Please specify a valid dataset " + ZSL_DATASETS.__str__()
    sources = target
    dataset = MatDataset
    configs['mixup_img_w'] = 0.0
    configs['iters_per_epoch'] = 'max'

configs['input_dim'] = input_dim
configs['semantic_w'] = semantic_w
configs['multi_domain'] = multi_domain

# Init loggers and checkpoints path
log_dir = args.log_dir
checkpoint_dir = args.ckpt_dir
exp_name = args.name
cudnn.benchmark = True


exp_name=args.name+'.pkl'


try:
    os.makedirs(log_dir)
except OSError:
    pass

try:
    os.makedirs(checkpoint_dir)
except OSError:
    pass

log_file = os.path.join(log_dir, exp_name)
if os.path.exists(log_file):
    print("WARNING: Your experiment logger seems to exist. Change the name to avoid unwanted overwriting.")

checkpoint_file = os.path.join(checkpoint_dir, args.name + '-runN.pth')
if os.path.exists(checkpoint_file):
    print("WARNING: Your experiment checkpoint seems to exist. Change the name to avoid unwanted overwriting.")

logger = {'results':[], 'config': configs, 'target': target, 'checkpoints':[], 'sem_loss':[[] for _ in range(args.runs)],
          'mimg_loss':[[] for _ in range(args.runs)], 'mfeat_loss':[[] for _ in range(args.runs)],
          'mratio_loss':[[] for _ in range(args.runs)]}
results = []
results_top = []
val_datasets = None

tr_dataset = dataset(args.data_root, sources,train=True)
"finding samples per class"
cls_lbl = tr_dataset.labels
#tr_dataset_class = [x[3] for x in tr_dataset]

cls_lbl = cls_lbl.tolist()
samples_per_class = Counter(cls_lbl)
print("no of train samples:", sum(samples_per_class.values()))
#import pdb;pdb.set_trace()
tst_dataset = dataset(args.data_root, target, train=False)
tst_cls_lbl = tst_dataset.labels
#tr_dataset_class = [x[3] for x in tr_dataset]

tst_cls_lbl = tst_cls_lbl.tolist()
tst_samples_per_class = Counter(tst_cls_lbl)
print("no of test samples:", sum(tst_samples_per_class.values()))
# Start experiments loop
for r in range(args.runs):
    print('\nTarget: ' + target + '  run ' +str(r+1)+'/'+str(args.runs))

    # Create datasets

    train_dataset = dataset(args.data_root, sources,train=True)
    #cls_lbl = torch.tensor(len(train_dataset))

    #train_dataset_class = [x[:][3].item() for x in train_dataset]   #"---newly added"
    test_dataset = dataset(args.data_root, target, train=False)
    #import pdb;pdb.set_trace()
    if args.dg and not args.zsl:
        val_datasets = []
        for s in sources:
            val_datasets.append(dataset(args.data_root, s, train=False, validation=True))

    attributes = train_dataset.full_attributes
    seen = train_dataset.seen
    unseen = train_dataset.unseen

    # Init method
    method = CuMix(seen_classes=seen,unseen_classes=unseen,attributes=attributes,configs=configs, samples_per_class = samples_per_class,
                   tst_samples_per_class = tst_samples_per_class, zsl_only = not args.dg,
                   dg_only = not args.zsl, device=device, world_size=world_size, rank=rank)

    temp_results = []
    top_sources = 0.
    top_idx=-1
    epoch = configs['epochs']
    #import pdb;pdb.set_trace()
    # Strat training loop
    pth = os.path.join("saved_embeddings","{}".format(target), "{}epochs".format(epoch))
    save_embed_path = os.path.join(args.data_root, pth)
    if not os.path.isdir(save_embed_path):
        os.makedirs(save_embed_path)
        print("creating Save Embedding directory")
    else:
        print("Save Embedding directory exists")

    # import pdb;
    # pdb.set_trace()

    for e in tqdm(range(0, configs['epochs'])):

        pth1 = os.path.join(save_embed_path, "test_embedding_epoch{}_run{}.npy".format(e, r))
        pth2 = os.path.join(save_embed_path, "test_semantic_embedding_epoch{}_run{}.npy".format(e, r))

        #import pdb;pdb.set_trace()
        semantic_loss, mimg_loss, mfeat_loss, mratio_loss = method.fit(train_dataset, e)
        accuracy, test_fix_embed, test_embeddings, target_pred, overall_accuracy = test(method, test_dataset, attributes, tst_cls_lbl, zsl=args.zsl)

        save(pth1, test_embeddings)
        if e == configs['epochs']-1:
            pth2 = os.path.join(save_embed_path, "test_semantic_embedding_epoch{}_run{}.npy".format(e, r))
            save(pth2, test_fix_embed)
            pth3 = os.path.join(save_embed_path, "test_target_epoch{}_run{}.npy".format(e, r))
            save(pth3, target_pred)
        # In case of DG only, perform validation on source domains, as in https://arxiv.org/pdf/1710.03077.pdf

        if val_datasets is not None:
            acc_sources = 0.
            for v in val_datasets:
                acc_sources += test(method, v, device, zsl=False)
            acc_sources /= len(sources)
            if acc_sources > top_sources:
                top_sources = acc_sources
                temp_results = accuracy

        else:
            temp_results = accuracy

        # Store losses
        logger['sem_loss'][r].append(semantic_loss)
        logger['mimg_loss'][r].append(mimg_loss)
        logger['mfeat_loss'][r].append(mfeat_loss)
        logger['mratio_loss'][r].append((mratio_loss))
        print(".........Test accuracy is {}".format(temp_results))
        print(".........Test overall accuracy is {}".format(overall_accuracy))
        print("sem loss:{:.4f} mimg_loss:{:.4f} mfeat_loss:{:.4f} mratio_loss:{:.4f}"
                        .format(semantic_loss, mimg_loss, mfeat_loss, mratio_loss))
    # Store checkpoints and update logger
    checkpoint_dict = {}
    method.save(checkpoint_dict)
    current_checkpoint_name = checkpoint_file.replace('runN.pth','run'+str(r+1)+'.pth')
    torch.save(checkpoint_dict, current_checkpoint_name)

    logger['results'].append(temp_results)
    logger['checkpoints'].append(current_checkpoint_name)
    print(target,logger['results'][top_idx])


print('\nResults for ' + target, np.mean(logger['results']),np.std(logger['results']))

with open(log_file, 'wb') as handle:
    pickle.dump(logger, handle, protocol=pickle.HIGHEST_PROTOCOL)
