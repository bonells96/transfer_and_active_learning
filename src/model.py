import torch
import torch.nn.functional as F 
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinearNet, self).__init__()

        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        self.hidden = nn.Linear(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        
        embeddings = self.embeddings.encode(x, convert_to_tensor=True)
        out = F.relu(self.hidden(embeddings))
        out = torch.sigmoid(self.fc(out))

        return out
    
    def predict(self, x):
        
        out = self.forward(x)
        _,pred = torch.max(out, 1)
        return pred 


def train_one_epoch(model, train_loader, optimizer):

    running_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        #embeddings = model_embeddings.encode(inputs, convert_to_tensor=True)
        output = model(inputs).squeeze()
        target = F.one_hot(labels, num_classes=2).to(torch.float32)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss
