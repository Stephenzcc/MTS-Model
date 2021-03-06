class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 2
        self.kernel_size = 3
        self.stride = 1
        self.final_out_channels = 128
        self.dilations = [32, 64, 64, 64]

        self.d_input = 8
        self.d_model = 2
        self.d_output = 128
        self.q = 4
        self.v = 4
        self.h = 4
        self.N = 4
        self.attention_size = 3
        self.chunk_mode = None
        self.pe = "regular"

        self.num_classes = 10
        self.dropout = 0.35
        self.features_len = 4

        # training configs
        self.num_epoch = 80

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-3

        # data parameters
        self.drop_last = True
        self.batch_size = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 2


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 2
