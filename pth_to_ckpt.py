from source.model import GNNLightning
import torch
import pytorch_lightning as pl

# Carica il file originale
state = torch.load("/home/palu001/Github/Deep-Learning-Hackaton/model_A_best_f1.pth", map_location="cpu")

# Estrai il dizionario dei pesi
state_dict = state["state_dict"] if "state_dict" in state else state

# Aggiungi il prefisso "model." alle chiavi
new_state_dict = {f"model.{k}": v for k, v in state_dict.items()}

# Istanzia il modello
model = GNNLightning(gnn='gin-virtual', dataset_name='B', loss_n =1, num_layer = 5, emb_dim = 300, drop_ratio = 0.5)

# Carica i pesi modificati nel modello
model.load_state_dict(new_state_dict)

# Salva il nuovo checkpoint .ckpt
checkpoint_path = "/home/palu001/Github/Deep-Learning-Hackaton/model_A_best.ckpt"
torch.save({
    'state_dict': model.state_dict(),
    'pytorch-lightning_version': pl.__version__,
    'hyper_parameters': model.hparams if hasattr(model, "hparams") else {},
}, checkpoint_path)
