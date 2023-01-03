import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from data.domain_dataset import DistributedBalancedSampler,BalancedSampler
from torchvision.models.resnet import resnet18,resnet50
from collections import Counter
from collections import OrderedDict
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import itertools
import random
#from numpy import save
import os
# Init losses and utility functions for mixing samples
CE = nn.CrossEntropyLoss()
RG = np.random.default_rng()

"Reproducibility"
# np.random.seed(0)
# random.seed(0)
# #torch.random.seed(0)
# torch.random.manual_seed(0)

def swap(xs, a, b):
    xs[a], xs[b] = xs[b], xs[a]

def derange(xs):
    x_new = [] + xs
    for a in range(1, len(x_new)):
        b = RG.choice(range(0, a))
        swap(x_new, a, b)
    return x_new

# CE on mixed labels, represented as vectors
def manual_CE(predictions, labels):
    loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions, dim=1), dim=1))
    return loss


# Standard mix
def std_mix(x,indeces,ratio):
    return ratio*x + (1.-ratio)*x[indeces]

# Init ZSL classifier with normalized embeddings
class UnitClassifier(nn.Module):
    def __init__(self, attributes, fc_margin, classes, samples_per_class, device='cuda', train = True):
        super(UnitClassifier, self).__init__()

        self.attributes = attributes
        self.device = device
        self.classes = classes
        self.cos = nn.CosineSimilarity(dim=1)
        self.fc = nn.Linear(attributes[0].size(0), classes.size(0), bias=False).to(device)
        if train:
            "Class imbalance handling"
            self.samples_per_class = samples_per_class
            noise = np.random.normal(0, 1, len(classes))
            noise = torch.from_numpy(noise).float().to(device)
            self.noise = noise
            samples_value = self.samples_per_class.values()
            total_samples = sum(samples_value)
        else:
            pass
        self.fc_margin = fc_margin
        for i,c in enumerate(classes):
            if train:
                attributes[c.item()] = attributes[c.item()].to(device)
                norm_attributes = attributes[c.item()]

            else:
                norm_attributes = attributes[c.item()].to(device)
            norm_attributes/=torch.norm(norm_attributes,2)
            self.fc.weight[i].data[:] = norm_attributes

    def forward(self, x, labels, trn):

        if trn:
            noise_batch = self.noise[labels]
            margin_batch = self.fc_margin[labels]
            margin_batch = margin_batch.to(self.device)
            apsln = noise_batch * margin_batch
            mu = apsln.unsqueeze(1) * x
            o = self.fc(x)
            score = o + mu
            return score, self.fc_margin
        else:
            o = self.fc(x)
            return o

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        try:
            m.bias.data.fill_(0)
        except:
            print('bias not present')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Actual method
class CuMix:

    # Init following the given config
    def __init__(self, seen_classes, unseen_classes, attributes, configs, samples_per_class, tst_samples_per_class,
                 zsl_only=False, dg_only=False, device='cuda', world_size=1, rank=0):
        self.end_to_end = True
        self.domain_mix = True
        self.samples_per_class = samples_per_class
        self.tst_samples_per_class = tst_samples_per_class

        if configs['backbone'] == 'none':
            self.end_to_end = False
            self.backbone = nn.Identity()
            self.lr_net = None
        else:
            backbone = eval(configs['backbone'])
            self.backbone = backbone(pretrained=True)
            self.backbone.fc = nn.Identity()
            self.lr_net=configs['lr_net']
            self.backbone.to(device)
            if world_size>1:
                #self.backbone = self.backbone.to(device)
                self.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
                #newly added
                self.backbone = torch.nn.parallel.DistributedDataParallel(self.backbone,
                                                        device_ids = [id], output_device = id)

            self.backbone.eval()

        self.zsl_only = zsl_only
        if self.zsl_only:
            self.domain_mix = False

        self.seen_classes = seen_classes if not dg_only else torch.Tensor([0 for _ in range(seen_classes)])
        self.unseen_classes = unseen_classes
        self.attributes = attributes
        self.rank = rank
        self.world_size = world_size

        self.device = device

        attSize = 0 if dg_only else self.attributes[0].size(0)
        self.current_epoch = -1
        self.mixup_w = configs['mixup_img_w']
        self.mixup_feat_w = configs['mixup_feat_w']

        self.max_beta = configs['mixup_beta']
        self.mixup_beta = 0.0
        self.mixup_step = configs['mixup_step']

        self.step = configs['step']
        self.batch_size = configs['batch_size']

        self.lr = configs['lr']
        # import pdb;
        # pdb.set_trace()
        self.nesterov = configs['nesterov']
        self.decay = configs['weight_decay']
        self.freeze_bn = configs['freeze_bn']

        input_dim = configs['input_dim']
        self.semantic_w = configs['semantic_w']

        if dg_only:
            self.semantic_projector = nn.Identity()

            self.train_classifier = nn.Linear(input_dim, unseen_classes)
            self.train_classifier.apply(weights_init)
            self.train_classifier = self.train_classifier.to(self.device)
            self.train_classifier.eval()
            self.final_classifier = self.train_classifier
        else:
            self.semantic_projector = nn.Linear(input_dim, attSize)
            self.semantic_projector.apply(weights_init)
            self.semantic_projector = self.semantic_projector.to(self.device)
            self.semantic_projector.eval()

            "Adaptive Margin Generator"
            total_samples = sum(self.samples_per_class.values())
            dtype = torch.cuda.FloatTensor
            margin = [(total_samples-samples_per_class[c.item()])/total_samples for i,c in enumerate(seen_classes)]
            self.fc_margin = nn.Parameter(torch.tensor(margin), requires_grad=True)
            self.train_classifier = UnitClassifier(self.attributes, self.fc_margin, seen_classes, self.samples_per_class, self.device, train = True)
            self.train_classifier.eval()
            self.final_classifier = UnitClassifier(self.attributes, self.fc_margin, unseen_classes, self.tst_samples_per_class, self.device, train = False)
            self.final_classifier.eval()

            "mix ratio"
            self.mixratio_pred = nn.Linear(input_dim, len(seen_classes)).to(self.device)


        if not configs['multi_domain']:
            self.dpb = 1
            self.iters = None
        else:
            self.dpb = configs['domains_per_batch']
            self.iters = configs['iters_per_epoch']


        self.criterion = CE
        self.mixup_criterion = manual_CE
        self.current_epoch = -1
        self.dg_only = dg_only

    # Create one hot labels
    def create_one_hot(self, y):
        y_onehot = torch.LongTensor(y.size(0), self.seen_classes.size(0)).to(self.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot

    # Utilities for saving/loading/retrieving parameters
    def get_classifier_params(self):
        if self.dg_only:
            return self.train_classifier.parameters()
        return self.semantic_projector.parameters()

    def save(self, dict):
        dict['backbone'] = self.backbone.state_dict()
        dict['semantic_projector'] = self.semantic_projector.state_dict()
        dict['train_classifier'] = self.train_classifier.state_dict()
        dict['final_classifier'] = self.final_classifier.state_dict()
        dict['epoch'] = self.current_epoch
        dict['mixratio_pred'] = self.mixratio_pred.state_dict()

    def load(self, dict):
        self.backbone.load_state_dict(dict['backbone'])
        self.semantic_projector.load_state_dict(dict['semantic_projector'])
        self.train_classifier.load_state_dict(dict['train_classifier'])
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier.load_state_dict(dict['final_classifier'])
        try:
            self.current_epoch = dict['epoch']
        except:
            self.current_epoch = 0
        self.mixratio_pred.load_state_dict((dict['mixratio_pred']))

    def to(self, device, parallel, id=0):  # for Cumix id = 0
        self.backbone = self.backbone.to(device)
        self.semantic_projector = self.semantic_projector.to(device)
        self.train_classifier = self.train_classifier.to(device)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = self.final_classifier.to(device)
        self.mixratio_pred = self.mixratio_pred.to(device)
        self.fc_margin = self.fc_margin.to(device)
        #import pdb;pdb.set_trace()
        if parallel:
            self.backbone = DistributedDataParallel(self.backbone, device_ids=[id], output_device=id)
            self.semantic_projector = DistributedDataParallel(self.semantic_projector, device_ids=[id],
                                                              output_device=id)
            self.train_classifier = DistributedDataParallel(self.train_classifier, device_ids=[id], output_device=id)
            if self.dg_only:
                self.final_classifier = self.train_classifier
            else:
                self.final_classifier = DistributedDataParallel(self.final_classifier, device_ids=[id], output_device=id)
            self.mixratio_pred = DistributedDataParallel(self.mixratio_pred, device_ids=[id], output_device=id)
            self.fc_margin = DistributedDataParallel(self.fc_margin, device_ids=[id], output_device=id)
        self.device = device

    # Utilities for going from train to eval mode
    def train(self):
        self.backbone.train()
        self.semantic_projector.train()
        self.train_classifier.train()
        self.final_classifier.train()
        self.mixratio_pred.train()

    def eval(self):
        self.backbone.eval()
        self.semantic_projector.eval()
        self.final_classifier.eval()
        self.train_classifier.eval()
        self.mixratio_pred.eval()
        #self.fc_margin.eval()

    def zero_grad(self):
        self.backbone.zero_grad()
        self.semantic_projector.zero_grad()
        self.final_classifier.zero_grad()
        self.train_classifier.zero_grad()
        self.mixratio_pred.zero_grad()

    # Utilities for forward passes
    def predict(self, input, labels):
        features = self.backbone(input)
        return self.final_classifier(self.semantic_projector(features), labels, trn = False), self.semantic_projector(features)

    def forward(self, input, labels, return_features=False):

        features = self.backbone(input)

        prediction, margin = self.train_classifier(self.semantic_projector(features), labels, trn = True)
        mixratio = self.mixratio_pred(features)
        if return_features:
            return prediction, features, mixratio, margin
        return prediction, mixratio

    def forward_features(self,features, labels):
        mixup_features_predictions, margin =self.train_classifier(self.semantic_projector(features), labels, trn = True)
        return mixup_features_predictions, self.mixratio_pred(features)

    # Get indeces of samples to mix, following the sampling distribution of the current epoch, as for the curriculum
    def get_sample_mixup(self, domains):
        # Check how many domains are in each batch (if 1 skip, but we have 1 just in ZSL only exps)
        if self.dpb>1:
            doms = list(range(len(torch.unique(domains))))
            bs = domains.size(0) // len(doms)
            selected = derange(doms)
            permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i] * bs) for i in range(len(doms))])
            permuted_within_dom = torch.cat([(torch.randperm(bs) + i * bs) for i in range(len(doms))])
            ratio_within_dom = torch.from_numpy(RG.binomial(1, self.mixup_domain, size=domains.size(0)))
            indeces = ratio_within_dom * permuted_within_dom + (1. - ratio_within_dom) * permuted_across_dom
        else:
            indeces = torch.randperm(domains.size(0))
        return indeces.long()

    # Get ratio to perform mixup
    def get_ratio_mixup(self,domains):
        return torch.from_numpy(RG.beta(self.mixup_beta, self.mixup_beta, size=domains.size(0))).float()

    # Get both
    def get_mixup_sample_and_ratio(self,domains):
        return self.get_sample_mixup(domains), self.get_ratio_mixup(domains)

    # Get mixed inputs/labels
    def get_mixed_input_labels(self,input,labels,indeces, ratios,dims=2):
        if dims==4:
            return std_mix(input, indeces, ratios.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))
        else:
            return std_mix(input, indeces, ratios.unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))

    def soft_cce(self, y_pred, y_true):
        loss = -torch.sum(y_true * torch.log_softmax(y_pred, dim=1), dim=1)
        return loss.mean()

    # Actual training procedure
    def fit(self, data, e):
        # Update mix related variables, as for the curriculum strategy
        self.current_epoch+=1
        self.mixup_beta = min(self.max_beta,max(self.max_beta*(self.current_epoch)/self.mixup_step,0.1))
        self.mixup_domain = min(1.0, max((self.mixup_step * 2.-self.current_epoch) / self.mixup_step, 0.0))

        # Init dataloaders
        if self.dpb>1:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=2,
                                sampler= DistributedBalancedSampler(data,self.batch_size//self.dpb,
                                                                   num_replicas=self.world_size, rank=self.rank,
                                                                   iters=self.iters,domains_per_batch=self.dpb),
                                drop_last=True)  # default num_workers = 8 drop_last = True
        else:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=8,
                                    shuffle=True, drop_last=True)

        # Init plus update optimizers
        if e<3:
            scale_lr = 1
        elif 2<e<7:
            scale_lr = 0.1
        else:
            scale_lr = 0.01

        print("Epoch:{} lr_net:{}".format(e, self.lr_net * scale_lr))
        optimizer_net = None

        if self.end_to_end:
                optimizer_net = optim.SGD(self.backbone.parameters(), lr=self.lr_net * scale_lr, momentum=0.9,
                                              weight_decay=self.decay, nesterov=self.nesterov)

        optimizer_zsl = optim.SGD(self.get_classifier_params(), lr=self.lr * scale_lr, momentum=0.9,
                                              weight_decay=self.decay, nesterov=self.nesterov)

        optimizer_mixratio = optim.SGD(self.mixratio_pred.parameters(), lr=self.lr * scale_lr, momentum=0.9,
                                          weight_decay=self.decay, nesterov=self.nesterov)

        optimizer_margin = optim.SGD([self.fc_margin], lr=self.lr * scale_lr, momentum=0.9,
                                     weight_decay=0, nesterov=self.nesterov)

        # Eventually freeze BN, done it for DG only
        if self.freeze_bn:
            self.eval()
        else:
            self.train()

        self.zero_grad()
        optimizer_margin.zero_grad()
        # Init logger values
        sem_loss = 0.
        mimg_loss = 0.
        mfeat_loss = 0.
        mratio_loss = 0
        img_mratio_loss = 0
        #import pdb;pdb.set_trace()
        for i, (inputs, _, domains, labels) in enumerate(dataloader):

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            one_hot_labels = self.create_one_hot(labels)

            #import pdb;pdb.set_trace()
            # Forward + compute AGG loss
            preds, features, mixratio, margin = self.forward(inputs, labels, return_features=True)
            if i == 0 and e==9:
                print("labels:", labels)
                print("margin:", margin)

            elif i == 0 and e==0:
                print("labels:", labels)
                print("margin:", margin)

            else:
                pass
            #     print(" margin.grad:", (self.fc_margin))
            semantic_loss = self.criterion(preds,labels)
            sem_loss += semantic_loss.item()

            # Forward on classifier + compute mixup loss on mixed features
            mix_indeces, mix_ratios = self.get_mixup_sample_and_ratio(domains)
            mix_ratios = mix_ratios.to(inputs.device)
            ratio_vec_gt = torch.zeros([inputs.size()[0], len(self.seen_classes)]).to(self.device)
            y_a = labels
            y_b = labels[mix_indeces]
            #import pdb;pdb.set_trace()
            for i in range(domains.size()[0]):
                ratio_vec_gt[i, y_a[i]] = mix_ratios[i]
                ratio_vec_gt[i, y_b[i]] = 1 - mix_ratios[i]
            mixup_features, mixup_labels = self.get_mixed_input_labels(features,one_hot_labels, mix_indeces,mix_ratios)

            mixup_features_predictions, mixup_ratio_pred = self.forward_features(mixup_features, labels)
            mixup_feature_loss = self.mixup_criterion(mixup_features_predictions, mixup_labels)
            mixupratio_loss = self.soft_cce(mixup_ratio_pred, ratio_vec_gt)
            total_loss= mixupratio_loss + self.semantic_w*semantic_loss+self.mixup_feat_w*mixup_feature_loss

            mfeat_loss += mixup_feature_loss.item()
            mratio_loss += mixupratio_loss.item()
            # Forward + compute mixup loss on mixed inputs, in case of end-to-end training
            # (skipped just for ZSL only exps)
            #import pdb;pdb.set_trace()
            if self.end_to_end:
                mix_indeces, mix_ratios = self.get_mixup_sample_and_ratio(domains)
                mixup_inputs, mixup_labels = self.get_mixed_input_labels(inputs,one_hot_labels, mix_indeces,mix_ratios.to(self.device),dims=4)
                mixup_img_predictions, img_mixup_ratio_pred = self.forward(mixup_inputs, labels, return_features=False)
                mixup_img_loss = self.mixup_criterion(mixup_img_predictions, mixup_labels)
                total_loss = total_loss + self.mixup_w*mixup_img_loss
                mimg_loss += mixup_img_loss.item()
            # Backward + update net
            self.zero_grad()
            optimizer_margin.zero_grad()
            total_loss.backward()
            if optimizer_net is not None:
                optimizer_net.step()
            optimizer_zsl.step()
            optimizer_mixratio.step()
            optimizer_margin.step()
            del total_loss
        self.eval()

        return sem_loss/((i+1)*self.batch_size), mimg_loss/((i+1)*self.batch_size), mfeat_loss/((i+1)*self.batch_size),\
               mratio_loss/((i+1)*self.batch_size),









