import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAggregator(nn.Module):
    def __init__(self, embedding_dim, num_documents=5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_documents = num_documents
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings):
        attention_weights = self.attention(embeddings)
        attention_weights = F.softmax(attention_weights, dim=1)
        aggregated_embedding = torch.sum(embeddings * attention_weights, dim=1)
        return aggregated_embedding


class BinaryClassificationModel(nn.Module):
    def __init__(self, embedding_dim, num_documents=5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_documents = num_documents
        self.attention_aggregator = AttentionAggregator(embedding_dim, num_documents)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, document_embeddings):
        document_embeddings = F.normalize(document_embeddings, p=2, dim=2)
        aggregated_embedding = self.attention_aggregator(document_embeddings)
        x = F.relu(self.fc1(aggregated_embedding))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        risk_score = torch.sigmoid(self.fc3(x))
        return risk_score.squeeze()

    def get_risk_prediction(self, risk_score):
        return risk_score
