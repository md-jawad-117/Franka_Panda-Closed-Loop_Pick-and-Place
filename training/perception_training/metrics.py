"""Foreground metrics that expose small targets and absent-target false positives."""
import numpy as np

NAMES=('background','capsule','fixture')


def confusion(pred,target):
    if pred.shape!=target.shape: raise ValueError('Prediction/label size mismatch')
    if np.any((pred<0)|(pred>2)|(target<0)|(target>2)): raise ValueError('Invalid semantic class')
    return np.bincount((target.ravel()*3+pred.ravel()).astype(np.int64),minlength=9).reshape(3,3)


def summarize(matrix):
    matrix=np.asarray(matrix); tp=np.diag(matrix); support=matrix.sum(1); predicted=matrix.sum(0)
    union=support+predicted-tp; result={}
    for i,name in enumerate(NAMES):
        result[name]={'iou':float(tp[i]/union[i]) if union[i] else None,
                      'dice':float(2*tp[i]/(support[i]+predicted[i])) if support[i]+predicted[i] else None,
                      'precision':float(tp[i]/predicted[i]) if predicted[i] else None,
                      'recall':float(tp[i]/support[i]) if support[i] else None,
                      'true_pixels':int(support[i]),'predicted_pixels':int(predicted[i])}
    fg=[result[name]['iou'] for name in NAMES[1:] if result[name]['iou'] is not None]
    return {'foreground_macro_iou':float(np.mean(fg)) if fg else None,'classes':result,'confusion':matrix.tolist()}


class MetricAccumulator:
    def __init__(self):
        self.matrix=np.zeros((3,3),np.int64);self.frames=0
        self.absent=np.zeros(3,np.int64);self.absent_fp=np.zeros(3,np.int64)
        self.absent_fp_pixels=np.zeros(3,np.int64);self.absent_total_pixels=np.zeros(3,np.int64)
    def update(self,pred,target):
        self.matrix+=confusion(pred,target);self.frames+=1
        for cls in (1,2):
            if not np.any(target==cls):
                self.absent[cls]+=1;self.absent_fp[cls]+=np.any(pred==cls)
                self.absent_fp_pixels[cls]+=int(np.count_nonzero(pred==cls));self.absent_total_pixels[cls]+=target.size
    def report(self):
        result=summarize(self.matrix);result['frames']=self.frames
        result['absent_target_frames']={NAMES[c]:{'frames':int(self.absent[c]),'with_any_false_positive_pixels':int(self.absent_fp[c]),
            'false_positive_pixels':int(self.absent_fp_pixels[c]),
            'false_positive_pixel_fraction':float(self.absent_fp_pixels[c]/self.absent_total_pixels[c]) if self.absent_total_pixels[c] else None} for c in (1,2)}
        return result
