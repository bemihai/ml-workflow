def get_cfg():
    """
    Get a copy of the default config.
    """
    from .defaults import _C
    return _C.clone()


def imagedata_args(cfg):
    """Image data args."""
    return {
        'root': cfg.data.root,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'use_gpu': cfg.use_gpu,
        'split': cfg.data.split,
        'train_batch_size': cfg.train.batch_size,
        'test_batch_size': cfg.test.batch_size,
        'num_workers': cfg.data.num_workers,
        'pin_memory': cfg.data.pin_memory,
        'train_sampler': cfg.sampler.train_sampler,
    }


def model_args(cfg):
    """Model args."""
    return {
        'model_name': cfg.model.name,
        'feature_dim': cfg.model.feature_dim,
        'pretrained': cfg.model.pretrained,
        'load_weights': cfg.model.load_weights,
        'resume': cfg.model.resume,
        'use_gpu': cfg.model.use_gpu,
    }


def optimizer_args(cfg):
    """Optimizer args."""
    return {
        'optim': cfg.train.optim,
        'lr': cfg.train.lr,
        'weight_decay': cfg.train.weight_decay,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
        'staged_lr': cfg.train.staged_lr,
        'new_layers': cfg.train.new_layers,
        'base_lr_mult': cfg.train.base_lr_mult
    }


def lr_scheduler_args(cfg):
    """Learning rate scheduler args."""
    return {
        'lr_scheduler': cfg.train.lr_scheduler,
        'stepsize': cfg.train.stepsize,
        'gamma': cfg.train.gamma,
        'max_epoch': cfg.train.max_epoch
    }


def trainer_args(cfg):
    return {
        'save_dir': cfg.data.save_dir,
        'use_gpu': cfg.use_gpu,
        'max_epoch': cfg.train.max_epoch,
        'start_epoch': cfg.train.start_epoch,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'open_layers': cfg.train.open_layers,
        'start_eval': cfg.test.start_eval,
        'eval_freq': cfg.test.eval_freq,
        'test_only': cfg.test.evaluate,
        'dist_metric': cfg.test.dist_metric,
        'normalize_feature': cfg.test.normalize_feature,
        'ranks': cfg.test.ranks,
        'es_monitor': cfg.train.es.monitor,
        'es_mode': cfg.train.es.mode,
        'es_verbose': cfg.train.es.verbose,
        'es_patience': cfg.train.es.patience,
    }
