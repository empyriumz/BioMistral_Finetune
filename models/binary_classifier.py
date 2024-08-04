import torch
import torch.nn as nn


class AttentionAggregator(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_documents=1,
        use_attention=True,
        use_time_weighting=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_documents = num_documents
        self.use_attention = use_attention
        self.use_time_weighting = use_time_weighting

        if self.use_attention and num_documents > 1:
            self.attention = nn.Linear(embedding_dim, 1)

        if self.use_time_weighting and num_documents > 1:
            self.gamma = nn.Parameter(torch.tensor([0.1]))

    def forward(self, embeddings, time_diffs=None):
        if embeddings.dim() == 2:  # Single document case
            return embeddings

        # Multi-document case
        valid_mask = (
            (time_diffs != -1).float().unsqueeze(-1)
            if time_diffs is not None
            else torch.ones_like(embeddings[:, :, 0]).unsqueeze(-1)
        )

        if self.use_attention:
            attention_weights = self.attention(embeddings)
            attention_weights = torch.softmax(attention_weights, dim=1)
        else:
            attention_weights = valid_mask / valid_mask.sum(dim=1, keepdim=True).clamp(
                min=1
            )

        if self.use_time_weighting and time_diffs is not None:
            decay_factor = torch.exp(-time_diffs.clamp(min=0) * self.gamma).unsqueeze(
                -1
            )
            combined_weights = attention_weights * decay_factor * valid_mask
        else:
            combined_weights = attention_weights * valid_mask

        combined_weights = combined_weights / combined_weights.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-8)
        aggregated_embedding = torch.sum(embeddings * combined_weights, dim=1)
        return aggregated_embedding


class BinaryClassificationModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_structured_features=0,
        num_documents=1,
        use_attention=True,
        use_time_weighting=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_documents = num_documents
        self.attention_aggregator = AttentionAggregator(
            embedding_dim,
            num_documents,
            use_attention=use_attention,
            use_time_weighting=use_time_weighting,
        )
        self.fc1 = nn.Linear(embedding_dim + num_structured_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, document_embeddings, time_diffs=None, structured_data=None):
        if document_embeddings.dim() == 2:
            # Single document case
            aggregated_embedding = document_embeddings
        else:
            # Multi-document case
            # Normalize only non-zero embeddings
            norm = torch.norm(document_embeddings, p=2, dim=2, keepdim=True)
            norm = torch.where(norm == 0, torch.ones_like(norm), norm)
            document_embeddings = document_embeddings / norm
            aggregated_embedding = self.attention_aggregator(
                document_embeddings, time_diffs
            )

        if structured_data is not None:
            combined_features = torch.cat(
                [aggregated_embedding, structured_data], dim=1
            )
        else:
            combined_features = aggregated_embedding

        x = self.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        risk_score = torch.sigmoid(self.fc3(x))
        return risk_score.squeeze()
