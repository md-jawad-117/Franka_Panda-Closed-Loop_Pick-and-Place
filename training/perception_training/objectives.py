"""Controlled alternatives to v1's pooled loss; inference inputs stay unchanged."""
import torch
from torch.nn import functional as F
from .model import segmentation_loss

OBJECTIVES=('legacy','image_class_balanced')
SELECTIONS=('global_foreground_iou','camera_macro_iou','view_foreground_iou')


def balanced_terms(logits,target):
    """Each image/class gets equal CE influence; foreground Dice treats absence.

    Present-class CE is averaged over that class's pixels before averaging over
    present classes and images. Foreground Dice is averaged per image/class.
    An absent foreground class instead gets mean -log(1-p) over the image, so
    background false positives are penalized explicitly. All arithmetic is FP32.
    """
    logits=logits.float()
    truth=F.one_hot(target,3).permute(0,3,1,2).float()
    counts=truth.sum((2,3));present=counts>0
    pixel_ce=F.cross_entropy(logits,target,reduction='none')
    ce_by_class=(pixel_ce[:,None]*truth).sum((2,3))/counts.clamp_min(1)
    ce=((ce_by_class*present).sum(1)/present.sum(1).clamp_min(1)).mean()
    probs=logits.softmax(1)
    overlap=(probs*truth).sum((2,3));denom=probs.sum((2,3))+counts
    dice=1-(2*overlap+1)/(denom+1)
    # Stable -log(1-p_class), including extremely confident false positives.
    log_total=torch.logsumexp(logits,dim=1)
    absent=torch.stack([log_total-torch.logsumexp(logits[:,[j for j in range(3) if j!=cls]],dim=1)
                        for cls in (1,2)],dim=1).mean((2,3))
    foreground=torch.where(present[:,1:],dice[:,1:],absent).mean()
    return ce,foreground


def loss_for(logits,target,weights,objective='legacy'):
    if objective=='legacy':return segmentation_loss(logits,target,weights)
    if objective=='image_class_balanced':
        ce,foreground=balanced_terms(logits,target)
        return ce+foreground
    raise ValueError(f'Unknown objective: {objective}')


def camera_macro_iou(by_camera):
    """Equal weight for wrist/top and capsule/fixture, independent of pixel area."""
    values=[]
    for camera in ('wrist_cam','top_cam'):
        if camera not in by_camera:raise ValueError(f'Validation lacks {camera}')
        for cls in ('capsule','fixture'):
            entry=by_camera[camera]['classes'][cls]
            if entry['true_pixels']==0:raise ValueError(f'Validation lacks true {camera}/{cls} pixels')
            values.append(entry['iou'])
    return sum(values)/len(values)


def selection_score(report,selection):
    if selection in ('global_foreground_iou','view_foreground_iou'):return report['foreground_macro_iou']
    if selection=='camera_macro_iou':return camera_macro_iou(report['by_camera'])
    raise ValueError(f'Unknown selection: {selection}')
