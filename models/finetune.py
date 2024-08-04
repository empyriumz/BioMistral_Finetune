import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType


def get_peft_config(lora_rank, lora_alpha, lora_dropout):
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "key", "value"],
        bias="none",
        modules_to_save=["classifier"],
    )


class LoRAGatortronClassifier(nn.Module):
    def __init__(
        self, model_name, num_labels=1, lora_rank=8, lora_alpha=32, lora_dropout=0.1
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # LoRA Configuration
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"],
        )

        # Apply LoRA to the model
        self.model = get_peft_model(self.model, peft_config)

        # Document-level attention
        self.doc_attention = nn.Linear(self.config.hidden_size, 1)

        # Classification head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, time_info):
        batch_size, num_docs, num_chunks, seq_length = input_ids.shape

        # Reshape input for processing
        input_ids = input_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)

        # Process through LoRA-modified model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get CLS token embeddings and reshape
        chunk_embeddings = outputs.last_hidden_state[:, 0].view(
            batch_size, num_docs, num_chunks, -1
        )

        # Average chunk embeddings to get document embeddings
        doc_embeddings = chunk_embeddings.mean(dim=2)

        # Calculate document attention weights
        doc_attention_weights = self.doc_attention(doc_embeddings).squeeze(-1)
        doc_mask = (time_info != -1).float()
        doc_attention_weights = doc_attention_weights.masked_fill(
            doc_mask == 0, float("-inf")
        )
        doc_attention_weights = torch.softmax(doc_attention_weights, dim=1)

        # Apply time-based weighting
        time_weights = torch.exp(-0.1 * time_info.clamp(min=0))
        doc_attention_weights = doc_attention_weights * time_weights * doc_mask
        doc_attention_weights = doc_attention_weights / doc_attention_weights.sum(
            dim=1, keepdim=True
        ).clamp(min=1e-9)

        # Weighted sum of document embeddings
        patient_embedding = torch.sum(
            doc_embeddings * doc_attention_weights.unsqueeze(-1), dim=1
        )

        # Classification
        logits = self.classifier(patient_embedding)
        return torch.sigmoid(logits).squeeze()
