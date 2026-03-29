import torch
from torch.utils.data import Dataset

class PairwiseDataset(Dataset):
    
    def __init__(self, pairs, features):
        self.pairs = pairs  # List of (i, j) tuples where i should rank above j
        self.features = torch.FloatTensor(features)
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.features[i], self.features[j]