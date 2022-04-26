import numpy as np
import torch
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def DataTransform(sample, config):

    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


    

def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
#     print(ExponentialSmoothing([-0.23653661,-0.098174649,0.3670918,1.3668598,0.49828124,-0.30022317,1.7156337,0.90210762,-0.21813165,-1.4307205,1.4413999,1.1249429,-0.12380626,-0.89113051,1.0509983,0.55146615,1.0104533,-1.3586705,-0.11208663,-0.46960289,-1.142277,-1.1801424,-0.70813653,-0.42765042,-0.018167225,0.40585047,-0.86945055,-0.4951533,1.21012,-1.2091844,-1.3875771,1.5823602,1.0215774,0.72071054,-1.3664114,-0.029589124,-1.3218877,-0.37259795,1.2793009,0.55723123,0.33228478,1.678066,0.47949736,-1.0836518,0.86911403,0.8655413,-1.4800079,1.199889,-0.20243331,-1.4948401,0.068904455,-1.4996038,0.40674365,1.289153,-1.1197309,-0.99598462,1.6511352,-1.1870984,0.025463346,-0.84151833]
# , trend="add", seasonal=None).fit().fittedvalues)
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

