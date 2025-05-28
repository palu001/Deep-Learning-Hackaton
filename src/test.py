import os
from src.dataset import GraphDataModule
from src.model import GNNLightning
import pytorch_lightning as pl

def test(test_path, model_type, batch_size, num_layers, embedding_dim, drop_ratio, loss_n, weight_decay):
    dataset_name = os.path.basename(os.path.dirname(test_path))
    print(f"Dataset name: {dataset_name}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    checkpoint_dir = os.path.join(project_root, "checkpoints", dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    dm = GraphDataModule(test_path=test_path, batch_size=batch_size)

    model = GNNLightning(gnn=model_type, num_layer=num_layers, emb_dim=embedding_dim, drop_ratio=drop_ratio, dataset_name=dataset_name, loss_n=loss_n, weight_decay=weight_decay)

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cuda",
        devices=1,
        logger=False
    )

    trainer.test(model, dm)

