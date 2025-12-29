# models/neural_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CreditScoringNN(nn.Module):
    """
    Нейронная сеть для кредитного скоринга
    Архитектура: 256-128-64-32-1 с BatchNorm и Dropout
    """
    def __init__(self, input_size, dropout_rate=0.3):
        super(CreditScoringNN, self).__init__()
        
        self.input_size = input_size
        
        self.layer1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.layer4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.layer3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.layer4(x)))
        x = torch.sigmoid(self.output(x))
        
        return x
    
    def get_num_parameters(self):
        """Возвращает количество параметров модели"""
        return sum(p.numel() for p in self.parameters())