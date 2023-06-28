class Config(object):
    def __init__(self):
        self.seq_length = 30
        # model configs
        self.input_channels = 6
        self.kernel_size = 4
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 4
        self.dropout = 0.35
        self.features_len = 3

        # training configs
        self.num_epoch = 60

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 16

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.2
        self.jitter_ratio = 0.8
        self.max_seg = 2


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 64
        self.timesteps = 1
