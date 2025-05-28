import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCELoss(torch.nn.Module):
    def __init__(self, q=0.7):
        super().__init__()
        assert q > 0 and q <= 1, "q must be in (0,1]"
        self.q = q

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        target_probs = probs[range(len(labels)), labels]
        loss = (1.0 - target_probs.pow(self.q)) / self.q
        return loss.mean()
        
class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, num_classes=6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, pred, labels):
        ce = self.ce(pred, labels)
        pred_soft = F.softmax(pred, dim=1).clamp(min=1e-7, max=1.0)
        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        rce = -torch.sum(pred_soft * torch.log(labels_one_hot + 1e-7), dim=1).mean()
        return self.alpha * ce + self.beta * rce

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy = 0.2):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()