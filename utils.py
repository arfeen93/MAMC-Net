import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import numpy as np
def test(model, test_data, attributes, tst_cls_lbl, device='cuda:1', zsl=True):
    model.eval()
    test_dataloader = DataLoader(test_data, batch_size=100, shuffle=False, num_workers=8, drop_last=False)

    # Performs average accuracy per class testing, as in ZSL
#if zsl:
    classes = test_data.classes
    test_feat = torch.zeros(0,)
    test_fix_embed = torch.zeros(0,).to(device)
    for c in set(tst_cls_lbl):
        test_fix_embed = torch.cat((test_fix_embed, torch.reshape(attributes[c], (1, attributes[c].shape[0]))), axis = 0)
    test_fix_embed = np.array(test_fix_embed.cpu())
    target_classes = torch.arange(classes)
    per_class_hits = torch.FloatTensor(classes).fill_(0).to(device)
    per_class_samples = torch.FloatTensor(classes).fill_(0).to(device)
    target_pred = []
    with torch.no_grad():
        for i, (input, _, doms, labels) in enumerate(test_dataloader):
            output, features = model.predict(input.to(device), labels.to(device))

            _, predicted_labels = torch.max(output.data, 1)
            for l in predicted_labels:
                target_pred.append(l.item())
            test_feat = torch.cat((test_feat, features.detach().cpu()), axis = 0)
            for tgt in target_classes:
                idx = (labels == tgt)
                if idx.float().sum() == 0:
                    continue
                per_class_hits[tgt] += torch.sum(labels[idx] == predicted_labels[idx].to('cpu'))
                per_class_samples[tgt] += torch.sum(idx).to('cpu')
    test_embeddings = np.array(test_feat)
    target_pred = np.array(target_pred)
    acc_per_class = per_class_hits / per_class_samples
    acc_unseen = acc_per_class.mean(0)

    # Overall accuracy
    hits = 0.
    samples = 0.

    with torch.no_grad():
        for i, (input, _, doms, labels) in enumerate(test_dataloader):
            output, feat = model.predict(input.to(device), labels)#.to('cpu')
            _, predicted_labels = torch.max(output.data, 1)
            hits += torch.sum(labels == predicted_labels.to('cpu')).item()  # Overall accuracy (not per class )
            samples += labels.size(0)

    return acc_unseen.item(), test_fix_embed, test_embeddings, target_pred, hits / samples
