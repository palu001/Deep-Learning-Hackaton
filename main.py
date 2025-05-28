import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from src.utils import set_seed
from src.test import test
from src.train import train

# Set the random seed
set_seed()


def main(args):
    if args.train_path is None:
        print("No training path provided. Skipping training.")
        test(args.test_path, args.gnn, args.batch_size, args.num_layer, args.emb_dim, args.drop_ratio, args.loss_n, args.weight_decay, args.val_size)

    else:
        print(f"Training on dataset: {args.train_path}")
        train(args.train_path, args.gnn, args.batch_size, args.epochs, args.num_layer, args.emb_dim, args.drop_ratio, args.loss_n, args.weight_decay, args.val_size, args.num_checkpoints)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, default = 5, help="Number of checkpoints to save during training.")
    parser.add_argument('--gnn', type=str, default='gin-virtual', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--loss_n', type=int, default=2, help="Loss: 1 (CE), 2 (Noisy CE), 3 (GCE), 4 (SCE) (default Noisy CE)")
    parser.add_argument('--weight_decay', type=bool, default=1, help='weight decay yes or no (default: 1)')
    parser.add_argument('--val_size', type=float, default=0.2, help='validation set size (default: 0.2)')
    args = parser.parse_args()
    main(args)