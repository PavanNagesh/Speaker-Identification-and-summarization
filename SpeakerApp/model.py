import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Define the Base Architecture (Matches your training) ---
class SpeakerEncoder(nn.Module):
    def __init__(self, num_speakers):
        super(SpeakerEncoder, self).__init__()
        
        # CNN Layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # GRU Layer
        self.gru_input_size = 128 * 10 
        self.gru_hidden_size = 512
        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden_size, 1, batch_first=True)

        # Fully Connected Layers
        self.embedding_dim = 768 
        self.fc1_embedding = nn.Linear(self.gru_hidden_size, self.embedding_dim)
        self.fc2_classifier = nn.Linear(self.embedding_dim, num_speakers)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.permute(0, 3, 2, 1) 
        x = x.reshape(x.size(0), x.size(1), -1) 
        
        _, hidden = self.gru(x)
        x = hidden.squeeze(0)

        x_embedding = F.relu(self.fc1_embedding(x))
        output_logits = self.fc2_classifier(x_embedding)
        return output_logits

# --- 2. Define the Extractor (The part we actually use) ---
class EmbeddingExtractor(SpeakerEncoder):
    def __init__(self, num_speakers):
        super(EmbeddingExtractor, self).__init__(num_speakers)

    def forward(self, x):
        # Same forward pass, but stops at the embedding
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.permute(0, 3, 2, 1) 
        x = x.reshape(x.size(0), x.size(1), -1) 
        
        _, hidden = self.gru(x)
        x = hidden.squeeze(0) 

        x_embedding = F.relu(self.fc1_embedding(x))
        return x_embedding