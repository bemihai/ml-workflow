import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.data_module import train_loader, val_loader
from models.classifier import OsnetClassifier


def main():

    mlflow.pytorch.autolog()
    mlflow.set_tracking_uri("http://localhost:5000/")

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs', save_top_k=1, verbose=True, monitor="val_loss", mode="min")
    
    model = OsnetClassifier()
    trainer = pl.Trainer(
        max_epochs=10,
        progress_bar_refresh_rate=20,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
    )

    with mlflow.start_run(run_name='test_3', experiment_id='1') as run:
        trainer.fit(model, train_loader, val_loader)
        trainer.test()
    
    
if __name__ == '__main__':
    main()
