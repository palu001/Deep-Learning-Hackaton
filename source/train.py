import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pytorch_lightning.callbacks import ModelCheckpoint
from source.dataset import GraphDataModule
from source.model import GNNLightning
import pytorch_lightning as pl

def train(train_path, model_type, batch_size, max_epochs, num_layers, embedding_dim, drop_ratio, loss_n, val_size, num_checkpoints):
    
    dataset_name = os.path.basename(os.path.dirname(train_path))
    print(f"Dataset name: {dataset_name}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    checkpoint_dir = os.path.join(project_root, "checkpoints", dataset_name)
    checkpoint_best = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    print(f"Training with model type: {model_type}, batch size: {batch_size}, max epochs: {max_epochs}, num layers: {num_layers}, embedding dim: {embedding_dim}, drop ratio: {drop_ratio}, loss type: {loss_n}, validation size: {val_size}, number of checkpoints: {num_checkpoints}")
    
    print(f"Dataset started loading")
    dm = GraphDataModule(train_path=train_path, batch_size=batch_size, val_split=val_size)
    print(f"Dataset loaded successfully")

    #I trained using torch (not lightning) so 
    checkpoint_callback_acc = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        dirpath=checkpoint_best,
        filename=f"model_{dataset_name}_best_acc",
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=True,
        verbose=True,
        auto_insert_metric_name=False
    )

    checkpoint_callback_f1 = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        dirpath=checkpoint_best,
        filename=f"model_{dataset_name}_best_f1",
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=True,
        verbose=True,
        auto_insert_metric_name=False
    )

    checkpoint_callback_epochs = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"model_{dataset_name}_epoch_{{epoch}}",
        save_top_k=-1,            
        save_last=False,
        every_n_epochs=max_epochs // num_checkpoints, 
        verbose=True,
        auto_insert_metric_name=False
    )

    print("Creating model...")
    model = GNNLightning(gnn= model_type, num_layer=num_layers, emb_dim=embedding_dim, drop_ratio=drop_ratio, dataset_name=dataset_name, loss_n=loss_n)
    print("Model created successfully")
    
    print("Starting training...")
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cuda",
        devices=1,
        callbacks=[checkpoint_callback_acc, checkpoint_callback_f1, checkpoint_callback_epochs],
        logger=False
    )
    
    trainer.fit(model, dm)
    print("Training completed successfully")