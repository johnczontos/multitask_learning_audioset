import torch
import torch.nn as nn

class MultiTaskGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes_list, dropout=0.5):
        super(MultiTaskGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Create task-specific networks
        self.task_nets = nn.ModuleList()
        for num_classes in num_classes_list:
            task_net = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
            self.task_nets.append(task_net)
    
    def forward(self, x, task_idx):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Use the last hidden state for classification
        out = out[:, -1, :]
        
        return self.task_nets[task_idx](out)

if __name__=="__main__":
    # Example usage
    input_size = 10  # Number of input features
    hidden_size = 64  # Number of features in hidden state
    num_layers = 2  # Number of GRU layers
    num_classes_list = [3, 5, 2]  # Number of classes for each task

    model = MultiTaskGRU(input_size, hidden_size, num_layers, num_classes_list)

    # Dummy input
    x = torch.randn(32, 50, input_size)  # (batch_size, sequence_length, input_size)

    # Forward pass
    output = model(x, 0)
    print(f"output shape: {output.shape}")
