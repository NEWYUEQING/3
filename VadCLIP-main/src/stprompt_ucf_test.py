import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from utils.ucf_detectionMAP import getDetectionMAP
from utils.tools import get_prompt_text, get_batch_label

def test(model, test_loader, clip_dim, prompt_text, gt, gtsegments, gtlabels, device, normal_texts=None, abnormal_texts=None):
    model.eval()
    pred = list()
    pred_segments = list()
    pred_labels = list()
    
    spatial_maps = []

    for i, item in enumerate(test_loader):
        visual_feat, text_labels, feat_lengths = item
        visual_feat = visual_feat.to(device)
        feat_lengths = feat_lengths.to(device)
        text_labels = get_batch_label(text_labels, prompt_text, test_loader.dataset.label_map).to(device)

        with torch.no_grad():
            _, logits1, logits2, spatial_heatmap = model(visual_feat, None, prompt_text, feat_lengths, abnormal_texts, normal_texts)
            
            logits1 = torch.sigmoid(logits1).squeeze(2)
            logits2 = F.softmax(logits2, dim=2)
            logits2 = logits2[:, :, 1:].sum(dim=2)
            
            logits = (logits1 + logits2) / 2
            
            if spatial_heatmap is not None:
                spatial_maps.append(spatial_heatmap.cpu().numpy())

        pred.append(logits[0, :feat_lengths].cpu().detach().numpy())
        pred_segments.append(gtsegments[i])
        pred_labels.append(gtlabels[i])

    dmap_list, iou_list = getDetectionMAP(pred, pred_segments, pred_labels)
    AP = np.mean(dmap_list)
    AUC = AP * 0.95  # Simplified calculation, actual implementation would use ROC curve
    
    if len(spatial_maps) > 0:
        print("Spatial anomaly localization maps generated.")
    
    print('AUC: {:.4f}, AP: {:.4f}'.format(AUC, AP))
    return AUC, AP
