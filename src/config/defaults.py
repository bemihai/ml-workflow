from fvcore.common.config import CfgNode

# main config node
_C = CfgNode()

# model
_C.model = CfgNode()
_C.model.name = ''
_C.model.feature_dim = 512  # output size for extracted features
_C.model.pretrained = False  # automatically load pretrained model weights if available
_C.model.load_weights = ''  # path to pretrained weights
_C.model.resume = ''  # path to checkpoint for resume training
_C.model.use_gpu = False

# data
_C.data = CfgNode()
_C.data.root = 'data'
_C.data.num_workers = 12  # number of data loading workers
_C.data.pin_memory = True
_C.data.transforms = []  # data transforms
_C.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
_C.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
_C.data.split = 0.05  # percent of data to keep for validation
_C.data.save_dir = 'outputs'  # dir to save outputs
_C.sampler = CfgNode()
_C.sampler.train_sampler = ''

# train
_C.train = CfgNode()
_C.train.optim = 'adam'
_C.train.lr = 0.0001
_C.train.weight_decay = 5e-4
_C.train.max_epoch = 50
_C.train.start_epoch = 0
_C.train.batch_size = 32
_C.train.fixbase_epoch = 0  # number of epochs to fix base layers
_C.train.open_layers = []  # layers for training while keeping others frozen
_C.train.staged_lr = False  # set different lr to different layers
_C.train.new_layers = []  # newly added layers with default lr
_C.train.base_lr_mult = 0.1  # learning rate multiplier for base layers
_C.train.lr_scheduler = 'single_step'
_C.train.stepsize = [20]  # stepsize to decay learning rate
_C.train.gamma = 0.1  # learning rate decay multiplier
_C.train.seed = 99  # random seed
_C.train.es = CfgNode()  # early stopping
_C.train.es.monitor = 'val_loss'  # monitor
_C.train.es.mode = 'min'  # mode
_C.train.es.verbose = 'True'  # verbose
_C.train.es.patience = 3  # patience

# optimizer
_C.sgd = CfgNode()
_C.sgd.momentum = 0.9  # momentum factor for sgd and respray
_C.sgd.dampening = 0.  # dampening for momentum
_C.sgd.nesterov = False  # Nesterov momentum
_C.rmsprop = CfgNode()
_C.rmsprop.alpha = 0.99  # smoothing constant
_C.adam = CfgNode()
_C.adam.beta1 = 0.9  # exponential decay rate for first moment
_C.adam.beta2 = 0.999  # exponential decay rate for second moment

# loss
_C.loss = CfgNode()
_C.loss.name = ''
_C.loss.softmax = CfgNode()
_C.loss.softmax.label_smooth = True  # use label smoothing regularizer
_C.loss.triplet = CfgNode()
_C.loss.triplet.margin = 0.3  # distance margin
_C.loss.triplet.weight_t = 1.  # weight to balance hard triplet loss
_C.loss.triplet.weight_x = 0.  # weight to balance cross entropy loss

# test
_C.test = CfgNode()
_C.test.batch_size = 100
_C.test.dist_metric = 'euclidean'  # distance metric, ['euclidean', 'cosine']
_C.test.normalize_feature = False  # normalize feature vectors before computing distance
_C.test.ranks = [1, 5, 10, 20]  # ranks
_C.test.evaluate = False  # test only
_C.test.eval_freq = -1  # evaluation frequency (-1 means to only test after training)
_C.test.start_eval = 0  # start to evaluate after a specific epoch
_C.test.metrics = []  # metrics
