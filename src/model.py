import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torchmetrics
from src.conv import GNN_node, GNN_node_Virtualnode
from src.loss import NoisyCrossEntropyLoss, SCELoss, GCELoss
import os
import pytorch_lightning as pl
import pandas as pd



class GNN(torch.nn.Module):
    
    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)
    
class GNNLightning(pl.LightningModule):
    def __init__(self, gnn, dataset_name, loss_n, weight_decay, num_layer, emb_dim, drop_ratio):
        super(GNNLightning, self).__init__()
        
        if gnn == 'gin':
            self.model = GNN(gnn_type = 'gin', num_class = 6, num_layer = num_layer, emb_dim = emb_dim, drop_ratio = drop_ratio, virtual_node = False)
        elif gnn == 'gin-virtual':
            self.model = GNN(gnn_type = 'gin', num_class = 6, num_layer = num_layer, emb_dim = emb_dim, drop_ratio = drop_ratio, virtual_node = True)
        elif gnn == 'gcn':
            self.model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = num_layer, emb_dim = emb_dim, drop_ratio = drop_ratio, virtual_node = False)
        elif gnn == 'gcn-virtual':
            self.model = GNN(gnn_type = 'gcn', num_class = 6, num_layer = num_layer, emb_dim = emb_dim, drop_ratio = drop_ratio, virtual_node = True)
        else:
            raise ValueError('Invalid GNN type')
        
        self.weight_decay = weight_decay
        
        if loss_n == 1:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_n == 2:
            self.loss_fn = NoisyCrossEntropyLoss()
        elif loss_n == 3:
            self.loss_fn = GCELoss(num_classes=6)
        else:
            self.loss_fn = SCELoss(num_classes=6)
        

        self.eval_metric = torchmetrics.Accuracy(task="multiclass", num_classes=6)
        self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=6, average='macro')

        self.dataset_name = dataset_name

        self.log_train = f"logs/{dataset_name}/train.log"
        self.log_val = f"logs/{dataset_name}/val.log"

        self.train_loss_list = []
        self.train_acc_list = []
        self.train_f1_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        self.val_f1_list = []
        self.test_predictions = []


    def forward(self, batched_data):
        output = self.model(batched_data)
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log('train_loss_step', loss, on_step=True, on_epoch = False, prog_bar=True)
        self.train_loss_list.append(loss.item())
        preds = torch.argmax(output, dim=1)
        acc = self.eval_metric(preds, batch.y)
        f1 = self.f1_metric(preds, batch.y)
        self.log('train_f1_step', f1, on_step=True, on_epoch = False, prog_bar=True)
        self.log('train_acc_step', acc, on_step=True, on_epoch = False, prog_bar=True)
        self.train_acc_list.append(acc.item())
        self.train_f1_list.append(f1.item())
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss_fn(output, batch.y)
        self.log('val_loss_step', loss, on_step=True, on_epoch = False, prog_bar=True)
        preds = torch.argmax(output, dim=1)
        acc = self.eval_metric(preds, batch.y)
        f1 = self.f1_metric(preds, batch.y)
        self.log('val_f1_step', f1, on_step=True, on_epoch = False, prog_bar=True)
        self.log('val_acc_step', acc, on_step=True, on_epoch = False, prog_bar=True)
        self.val_loss_list.append(loss.item())
        self.val_acc_list.append(acc.item())
        self.val_f1_list.append(f1.item())

    def on_train_epoch_end(self):
        avg_loss = sum(self.train_loss_list) / len(self.train_loss_list)
        self.log('train_loss', avg_loss, on_step=False, on_epoch = True, prog_bar=True)

        avg_acc = sum(self.train_acc_list) / len(self.train_acc_list)
        self.log('train_acc', avg_acc, on_step=False, on_epoch = True, prog_bar=True)

        avg_f1 = sum(self.train_f1_list) / len(self.train_f1_list)
        self.log('train_f1', avg_f1, on_step=False, on_epoch = True, prog_bar=True)

        self.train_f1_list = []
        self.train_acc_list = []
        self.train_loss_list = []

        os.makedirs(os.path.dirname(self.log_train), exist_ok=True)
        with open(self.log_train, 'a') as f:
            f.write(f"Epoch {self.current_epoch}: train_loss: {avg_loss}, train_acc: {avg_acc}, train_f1: {avg_f1}\n")
        
    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.val_loss_list) / len(self.val_loss_list)
        self.log('val_loss', avg_val_loss, on_step=False, on_epoch = True, prog_bar=True)
        self.val_loss_list = []

        avg_val_acc = sum(self.val_acc_list) / len(self.val_acc_list)
        self.log('val_acc', avg_val_acc, on_step=False, on_epoch = True, prog_bar=True)
        self.val_acc_list = []

        avg_val_f1 = sum(self.val_f1_list) / len(self.val_f1_list)
        self.log('val_f1', avg_val_f1, on_step=False, on_epoch = True, prog_bar=True)
        self.val_f1_list = []
        
        os.makedirs(os.path.dirname(self.log_val), exist_ok=True)
        if not self.trainer.sanity_checking:
            with open(self.log_val, 'a') as f:
                f.write(f"Epoch {self.current_epoch}: val_loss: {avg_val_loss}, val_acc: {avg_val_acc}, val_f1: {avg_val_f1}\n")
        
    def test_step(self, batch, batch_idx):
        output = self.forward(batch)
        preds = torch.argmax(output, dim=1)
        preds = preds.cpu().numpy().tolist()
        self.test_predictions.extend(preds)
    
        return preds
    
    def on_test_epoch_end(self):
        test_graph_ids = list(range(len(self.test_predictions)))  # Generate IDs for graphs

        output_df = pd.DataFrame({
            "id": test_graph_ids,
            "pred": self.test_predictions
        })
        output_csv_path = f"submission/testset_{self.dataset_name}.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        output_df.to_csv(output_csv_path, index=False)
        self.test_predictions = []

    def configure_optimizers(self):
        if self.weight_decay:
            optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-5)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer