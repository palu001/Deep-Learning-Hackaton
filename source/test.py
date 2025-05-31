import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from source.dataset import GraphDataModule
from source.model import GNNLightning
import pytorch_lightning as pl
import pandas as pd
import numpy as np

def test(test_path, model_type, batch_size, num_layers, embedding_dim, drop_ratio, loss_n, val_split):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = os.path.basename(os.path.dirname(test_path))
    print(f"Dataset name: {dataset_name}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{dataset_name}_best.ckpt")
    checkpoint_path_ensemble = os.path.join(checkpoint_dir, dataset_name)

    print("Loading dataset from:", test_path)
    dm = GraphDataModule(test_path=test_path, batch_size=batch_size, val_split=val_split)
    print("Dataset loaded successfully.")

    print("Loading model...")
    model = None
    ensemble_models = []
    
    if dataset_name in ["A", "C"]:
        model = GNNLightning.load_from_checkpoint(checkpoint_path, gnn=model_type, num_layer=num_layers,
                                                 emb_dim=embedding_dim, drop_ratio=drop_ratio,
                                                 dataset_name=dataset_name, loss_n=loss_n)
    elif dataset_name == "B":
        model = GNNLightning.load_from_checkpoint(checkpoint_path, gnn=model_type, num_layer=num_layers,
                                                 emb_dim=embedding_dim, drop_ratio=drop_ratio,
                                                 dataset_name=dataset_name, loss_n=loss_n)
        checkpoint_paths = [
            os.path.join(checkpoint_path_ensemble, f"model_{dataset_name}_best_acc(cos).ckpt"),
            os.path.join(checkpoint_path_ensemble, f"model_{dataset_name}_best_acc(gce).ckpt"),
            os.path.join(checkpoint_path_ensemble, f"model_{dataset_name}_best_acc(sce).ckpt"),
        ]
        for ckpt in checkpoint_paths:
            ensemble_models.append(GNNLightning.load_from_checkpoint(ckpt, gnn=model_type, num_layer=num_layers,
                                                                    emb_dim=embedding_dim, drop_ratio=drop_ratio,
                                                                    dataset_name=dataset_name, loss_n=loss_n))
    elif dataset_name == "D":
        model = GNNLightning.load_from_checkpoint(checkpoint_path, gnn=model_type, num_layer=num_layers,
                                                 emb_dim=embedding_dim, drop_ratio=drop_ratio,
                                                 dataset_name=dataset_name, loss_n=loss_n)
        checkpoint_paths = [
            os.path.join(checkpoint_path_ensemble, f"model_{dataset_name}_best_acc(cos).ckpt"),
            os.path.join(checkpoint_path_ensemble, f"model_{dataset_name}_best_acc(gce).ckpt"),
            os.path.join(checkpoint_path_ensemble, f"model_{dataset_name}_best_acc(sce).ckpt"),
            os.path.join(checkpoint_path_ensemble, f"model_{dataset_name}_best_f1(sce).ckpt"),
        ]
        for ckpt in checkpoint_paths:
            ensemble_models.append(GNNLightning.load_from_checkpoint(ckpt, gnn=model_type, num_layer=num_layers,
                                                                    emb_dim=embedding_dim, drop_ratio=drop_ratio,
                                                                    dataset_name=dataset_name, loss_n=loss_n))
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    print("Starting testing...")
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator=device,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    # Test main model
    print("Testing main model...")
    trainer.test(model, datamodule=dm)
    probs_list = [np.array(model.test_probabilities)]

    # Test ensemble models (if any)
    for i, ens_model in enumerate(ensemble_models):
        print(f"Testing ensemble model {i+1}...")
        trainer.test(ens_model, datamodule=dm)
        probs_list.append(np.array(ens_model.test_probabilities))

    # Somma delle probabilità di tutti i modelli caricati
    summed_probs = np.sum(probs_list, axis=0)
    final_preds = np.argmax(summed_probs, axis=1)

    # Crea DataFrame finale
    test_graph_ids = list(range(len(final_preds)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": final_preds
    })

    output_csv_path = f"submission/testset_{dataset_name}.csv"
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)

    print(f"Testing completed successfully. Predictions saved to {output_csv_path}")

