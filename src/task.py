import torch

class PatternDataset(torch.utils.data.Dataset):
    def __init__(self, frequencies, time):
        self.frequencies = frequencies
        self.time = time
        self.timeSeries = torch.arange(0, self.time)

    def __len__(self):
        return len(self.frequencies)

    def __getitem__(self, index):
        omega = self.frequencies[index]
        new_index = 100*omega - 9
        X = 0.25*torch.ones(self.lot) + new_index/51
        y = torch.unsqueeze(torch.sin(omega*self.timeSeries), dim=1)
        return X, y