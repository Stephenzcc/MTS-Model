class Config(object):
    def __init__(self):
        self.seq_length = 17984
        # model configs
        self.input_channels = 6
        self.kernel_size = 32
        self.stride = 8
        self.final_out_channels = 128

        self.num_classes = 5
        self.dropout = 0.35
        self.features_len = 142

        # training configs
        self.num_epoch = 30

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 4

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.01
        self.jitter_ratio = 0.01
        self.max_seg = 3


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 128
        self.timesteps = 60
