import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from stprompt_model import STPrompt
from stprompt_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label
import stprompt_option

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def contrastive_loss(text_features):
    loss = 0
    for i in range(text_features.shape[0]):
        for j in range(text_features.shape[0]):
            if i != j:
                text_i = text_features[i] / text_features[i].norm(dim=-1, keepdim=True)
                text_j = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                sim = torch.abs(text_i @ text_j)
                loss += torch.max(torch.tensor(0.0, device=text_features.device), sim)
    return loss / (text_features.shape[0] * (text_features.shape[0] - 1))

def train(model, train_loader, test_loader, args, label_map: dict, device):
    model.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    
    normal_texts = [
        "a picture of sky", "a picture of ground", "a picture of road", 
        "a picture of grass", "a picture of building", "a picture of wall", 
        "a picture of tree", "a picture of floor tile", "a picture of desk", 
        "a picture of cabinet", "a picture of chair", "a picture of door", 
        "a picture of blank"
    ]
    
    abnormal_texts = []
    for category in label_map.values():
        if category != "normal":
            abnormal_texts.append(f"a picture of {category}")
            if category == "fighting":
                abnormal_texts.extend(["people knockout someone", "people fighting", "violent behavior"])
            elif category == "car accident":
                abnormal_texts.extend(["people lying on the ground", "car crash", "vehicle collision"])
            elif category == "shooting":
                abnormal_texts.extend(["people shooting someone", "gun violence", "armed person"])
            elif category == "explosion":
                abnormal_texts.extend(["fire and smoke", "blast", "explosion aftermath"])
            elif category == "abuse":
                abnormal_texts.extend(["physical assault", "violent attack", "harmful behavior"])
            elif category == "riot":
                abnormal_texts.extend(["crowd violence", "public disorder", "violent protest"])
    
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        for i, item in enumerate(train_loader):
            step = 0
            visual_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device)
            feat_lengths = feat_lengths.to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2, _ = model(visual_feat, None, prompt_text, feat_lengths) 

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = contrastive_loss(text_features)
            loss_total3 += loss3.item()

            loss = loss1 + args.alpha * loss2 + args.beta * loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * train_loader.batch_size
            if step % 4800 == 0 and step != 0:
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), 
                      '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss_total3 / (i+1))
                
        scheduler.step()
        AUC, AP, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device, normal_texts, abnormal_texts)

        if AP > ap_best:
            ap_best = AP 
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, args.checkpoint_path)

        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = stprompt_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    train_dataset = XDDataset(args.visual_length, args.train_list, False, label_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = STPrompt(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
                     args.visual_head, args.visual_layers, args.attn_window, 
                     args.prompt_prefix, args.prompt_postfix, args.patch_size, 
                     args.stride, args.top_k, device)
    train(model, train_loader, test_loader, args, label_map, device)
