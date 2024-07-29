from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import odd_even_network, bitonic_network, execute_sort


class CustomDiffSortNet(torch.nn.Module):
    """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.
    Positional arguments:
    sorting_network_type -- which sorting network to use for sorting.
    vectors -- the matrix to sort along axis 1; sorted in-place
    Keyword arguments:
    steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
    art_lambda -- relevant for sigmoid_phi interpolation (default 0.25)
    interpolation_type -- how to interpolate when swapping two numbers; supported: `logistic`, `logistic_phi`,
                 (default 'logistic_phi')
    """

    def __init__(
        self,
        sorting_network_type: Literal["odd_even", "bitonic"],
        size: int,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        interpolation_type: str = None,
        distribution: str = "cauchy",
    ):
        super().__init__()
        self.sorting_network_type = sorting_network_type
        self.size = size

        # Register the sorting network in the module buffer.
        self._sorting_network_structure = self._setup_sorting_network_structure(
            sorting_network_type, size
        )
        self._register_sorting_network(self._sorting_network_structure)

        if interpolation_type is not None:
            assert (
                distribution is None
                or distribution == "cauchy"
                or distribution == interpolation_type
            ), (
                "Two different distributions have been set (distribution={} and"
                " interpolation_type={}); however, they have the same interpretation and"
                " interpolation_type is a deprecated argument".format(
                    distribution, interpolation_type
                )
            )
            distribution = interpolation_type

        self.steepness = steepness
        self.art_lambda = art_lambda
        self.distribution = distribution

    def forward(self, vectors):
        assert len(vectors.shape) == 2
        assert vectors.shape[1] == self.size
        sorted_out, predicted_permutation = self.sort(
            vectors, self.steepness, self.art_lambda, self.distribution
        )
        return sorted_out, predicted_permutation

    def _setup_sorting_network_structure(self, network_type, n):
        """Setup the sorting network structure. Used for registering the sorting network in the module buffer."""

        def matrix_to_torch(m):
            return [
                [torch.from_numpy(matrix).float() for matrix in matrix_set]
                for matrix_set in m
            ]

        if network_type == "bitonic":
            m = matrix_to_torch(bitonic_network(n))
        elif network_type == "odd_even":
            m = matrix_to_torch(odd_even_network(n))
        else:
            raise NotImplementedError(f"Sorting network `{network_type}` unknown.")

        return m

    def _register_sorting_network(self, m):
        """Register the sorting network in the module buffer."""
        for i, matrix_set in enumerate(m):
            for j, matrix in enumerate(matrix_set):
                self.register_buffer(f"sorting_network_{i}_{j}", matrix)

    def get_sorting_network(self):
        """Return the sorting network from the module buffer."""
        m = self._sorting_network_structure
        for i, _ in enumerate(m):
            yield (
                self.__getattr__(f"sorting_network_{i}_{j}") for j, _ in enumerate(m[i])
            )

    def sort(
        self,
        vectors: torch.Tensor,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        distribution: str = "cauchy",
    ):
        """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.

        Positional arguments:
        sorting_network
        vectors -- the matrix to sort along axis 1; sorted in-place

        Keyword arguments:
        steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
        art_lambda -- relevant for logistic_phi interpolation (default 0.25)
        distribution -- how to interpolate when swapping two numbers; (default 'cauchy')
        """
        assert self.sorting_network_0_0.device == vectors.device, (
            f"The sorting network is on device {self.sorting_network_0_0.device} while the vectors"
            f" are on device {vectors.device}, but they both need to be on the same device."
        )
        sorting_network = self.get_sorting_network()
        return execute_sort(
            sorting_network=sorting_network,
            vectors=vectors,
            steepness=steepness,
            art_lambda=art_lambda,
            distribution=distribution,
        )


class AttentionAggregator(nn.Module):
    def __init__(self, embedding_dim, num_documents=5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_documents = num_documents
        self.attention = nn.Linear(embedding_dim, 1)
        self.time_decay = nn.Parameter(torch.tensor(0.1))

    def forward(self, embeddings, timestamps):
        """
        Aggregate document embeddings using attention and time-based weighting.

        Args:
            embeddings (torch.Tensor): Document embeddings of shape (batch_size, num_documents, embedding_dim)
            timestamps (torch.Tensor): Timestamps for each document of shape (batch_size, num_documents)

        Returns:
            torch.Tensor: Aggregated embedding of shape (batch_size, embedding_dim)
        """
        # Input shape assertions
        assert (
            embeddings.shape[1] == self.num_documents
        ), f"Expected {self.num_documents} documents, but got {embeddings.shape[1]}"
        assert (
            embeddings.shape[2] == self.embedding_dim
        ), f"Expected embedding dimension of {self.embedding_dim}, but got {embeddings.shape[2]}"
        assert (
            timestamps.shape[1] == self.num_documents
        ), f"Expected {self.num_documents} timestamps, but got {timestamps.shape[1]}"

        attention_weights = self.attention(embeddings)  # (batch_size, num_documents, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Calculate time weights
        # Note: We use the first (most recent) timestamp as the reference
        latest_time = timestamps[:, 0].unsqueeze(1)
        time_diff = latest_time - timestamps
        time_weights = torch.exp(-self.time_decay * time_diff).unsqueeze(-1)

        # Combine attention and time weights
        combined_weights = attention_weights * time_weights
        combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)

        # Apply weights to embeddings
        aggregated_embedding = torch.sum(embeddings * combined_weights, dim=1)
        return aggregated_embedding


class SurvivalModel(nn.Module):
    def __init__(
        self, embedding_dim, structured_features, sorter_size=32, num_documents=5
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_documents = num_documents
        self.attention_aggregator = AttentionAggregator(embedding_dim, num_documents)
        self.fc = nn.Linear(embedding_dim + structured_features, sorter_size)
        self.sorter = CustomDiffSortNet(
            sorting_network_type="bitonic",
            size=sorter_size,
            steepness=2 * sorter_size,
            distribution="cauchy",
        )

    def forward(self, document_embeddings, timestamps, structured_data):
        """
        Forward pass of the survival model.

        Args:
            document_embeddings (torch.Tensor): Document embeddings of shape (batch_size, num_documents, embedding_dim)
            timestamps (torch.Tensor): Timestamps for each document of shape (batch_size, num_documents)
            structured_data (torch.Tensor): Structured patient data of shape (batch_size, structured_features)

        Returns:
            tuple: (logits, perm_prediction)
        """
        assert (
            document_embeddings.shape[1] == self.num_documents
        ), f"Expected {self.num_documents} documents, but got {document_embeddings.shape[1]}"
        assert (
            document_embeddings.shape[2] == self.embedding_dim
        ), f"Expected embedding dimension of {self.embedding_dim}, but got {document_embeddings.shape[2]}"

        aggregated_embedding = self.attention_aggregator(
            document_embeddings, timestamps
        )
        combined_features = torch.cat([aggregated_embedding, structured_data], dim=1)
        logits = self.fc(combined_features)
        _, perm_prediction = self.sorter(logits)
        return logits, perm_prediction

    def get_risk_scores(self, document_embeddings, timestamps, structured_data):
        aggregated_embedding = self.attention_aggregator(
            document_embeddings, timestamps
        )
        combined_features = torch.cat([aggregated_embedding, structured_data], dim=1)
        return self.fc(combined_features).squeeze(-1)
