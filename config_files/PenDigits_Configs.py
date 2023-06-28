# 序列长度太短，跳过3和4层
# x = self.conv_block2(x)
# x_r = self.conv_block5(x)
class Config(object):
    def __init__(self):
        self.seq_length = 8
        # model configs
        self.input_channels = 2
        self.kernel_size = 2
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 10
        self.dropout = 0.35
        self.features_len = 2

        # training configs
        self.num_epoch = 200 #100

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-3

        # data parameters
        self.drop_last = True
        self.batch_size = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.0 #0.5
        self.jitter_ratio = 0.5 #0.5
        self.max_seg = 2


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 1
