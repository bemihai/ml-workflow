from pytorch_lightning import Callback


class LoggerCallback(Callback):
    """
    Trainer logging callback. Logs trainer information to a provided custom logger.
    """

    def __init__(self, logger):
        self.logger = logger

    @staticmethod
    def _logged_metrics(trainer):
        metrics = "Logged metrics: \n"
        for key, val in trainer.logged_metrics.items():
            metrics += f"\t {key}: {val:.4f} \n"
        return metrics

    def on_train_start(self, trainer, pl_module):
        """Start training logs."""
        self.logger.info(f"Trainer set for {trainer.max_epochs} epochs starting from epoch {trainer.current_epoch}. \n"
                         f"Start training...")

    def on_test_start(self, trainer, pl_module):
        """Start testing logs."""
        self.logger.info("Start testing...")

    def on_test_end(self, trainer, pl_module):
        """End testing logs."""
        self.logger.info("Testing is done.")
        self.logger.info(self._logged_metrics(trainer))

    def on_train_end(self, trainer, pl_module):
        """End training logs."""
        self.logger.info("Training is done.")
        self.logger.info(self._logged_metrics(trainer))
        self.logger.info(f"Checkpoints saved to '{trainer.checkpoint_callback.dirpath}'")
        self.logger.info(f"Best model path: '{trainer.checkpoint_callback.best_model_path}' \n "
                         f"Last model path: '{trainer.checkpoint_callback.last_model_path}'")

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self.logger.info(f"Loaded {pl_module} from checkpoint at epoch {trainer.current_epoch}")
