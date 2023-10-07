import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, in_size, out_size, hidden_d, device,):
        super(VanillaRNN, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_d = hidden_d
        self.device = device
        
        self.rnn = nn.RNN(
            self.in_size, 
            self.hidden_d, 
            1, 
            nonlinearity='tanh', 
            batch_first = True, 
            bias=False
        ).to(device)
        
        self.out_layer = nn.Linear(
            self.hidden_d, 
            self.out_size, 
            bias=False
        ).to(device)
    
    def forward(self, x, hidden):
        rnn_out, hidden = self.rnn(x, hidden)
        out = self.out_layer(rnn_out)

        return out, rnn_out

    def init_hidden(self, batch_size):
        return torch.ones(1, batch_size, self.hidden_d).to(self.device)