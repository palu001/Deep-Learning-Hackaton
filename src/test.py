import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import os
from src.dataset import GraphDataModule
from src.model import GNNLightning
import pytorch_lightning as pl

def test(test_path, model_type, batch_size, num_layers, embedding_dim, drop_ratio, loss_n, weight_decay, val_split):

    dataset_name = os.path.basename(os.path.dirname(test_path))
    print(f"Dataset name: {dataset_name}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{dataset_name}_best.ckpt")

    print("Loading dataset from:", test_path)
    dm = GraphDataModule(test_path=test_path, batch_size=batch_size, val_split=val_split)
    print("Dataset loaded successfully.")

    print("Loading model from:", checkpoint_path)
    model = GNNLightning.load_from_checkpoint(checkpoint_path, gnn=model_type, num_layer=num_layers, emb_dim=embedding_dim, drop_ratio=drop_ratio, dataset_name=dataset_name, loss_n=loss_n, weight_decay=weight_decay)
    print("Model loaded successfully.")

    print("Starting testing...")
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cuda",
        devices=1,
        logger=False
    )

    trainer.test(model, dm)
    print("Testing completed successfully.")

