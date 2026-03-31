import torch
import torch.nn as nn
import numpy as np

class MagnitudeRegressionLoss(nn.Module):
    
    def __init__(self, scale=1.0, epsilon=1e-6):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred  
        batch_size = errors.shape[0]
        zero_vec = torch.zeros(1, errors.shape[1], device=errors.device)
        E = torch.cat([errors, zero_vec], dim=0)  
        diff = E.unsqueeze(0) - E.unsqueeze(1)
        distances = torch.norm(diff, dim=2, p=2)  
        zeta = torch.exp(-self.scale * distances)
        
        zeta = zeta + self.epsilon * torch.eye(zeta.shape[0], device=zeta.device)
        
        try:
            ones = torch.ones(zeta.shape[0], 1, device=zeta.device)
            w = torch.linalg.solve(zeta, ones)
            magnitude = torch.sum(w)
        except RuntimeError:
            magnitude = torch.tensor(float(batch_size + 1), device=y_pred.device)
        
        loss = magnitude - 1.0
        
        return loss