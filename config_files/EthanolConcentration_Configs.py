class Config(object):
    def __init__(self):
        self.seq_length = 1751
        # model configs
        self.input_channels = 3
        self.kernel_size = 16
        self.stride = 4
        self.final_out_channels = 256

        self.num_classes = 4
        self.dropout = 0.25
        self.features_len = 29 #17

        # training configs
        self.num_epoch = 45

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
        self.jitter_scale_ratio = 1.3
        self.jitter_ratio = 1.3
        self.max_seg = 50


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 256
        self.timesteps = 15
