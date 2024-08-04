import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal
from models.binary_classifier import AttentionAggregator


def bitonic_network(n):
    IDENTITY_MAP_FACTOR = 0.5
    num_blocks = math.ceil(np.log2(n))
    assert n <= 2**num_blocks
    network = []

    for block_idx in range(num_blocks):
        for layer_idx in range(block_idx + 1):
            m = 2 ** (block_idx - layer_idx)

            split_a, split_b = np.zeros((n, 2**num_blocks)), np.zeros(
                (n, 2**num_blocks)
            )
            combine_min, combine_max = np.zeros((2**num_blocks, n)), np.zeros(
                (2**num_blocks, n)
            )
            count = 0

            for i in range(0, 2**num_blocks, 2 * m):
                for j in range(m):
                    ix = i + j
                    a, b = ix, ix + m

                    # Cases to handle n \neq 2^k: The top wires are discarded and if a comparator considers them, the
                    # comparator is ignored.
                    if a >= 2**num_blocks - n and b >= 2**num_blocks - n:
                        split_a[count, a], split_b[count, b] = 1, 1
                        if (ix // 2 ** (block_idx + 1)) % 2 == 1:
                            a, b = b, a
                        combine_min[a, count], combine_max[b, count] = 1, 1
                        count += 1
                    elif a < 2**num_blocks - n and b < 2**num_blocks - n:
                        pass
                    elif a >= 2**num_blocks - n and b < 2**num_blocks - n:
                        split_a[count, a], split_b[count, a] = 1, 1
                        combine_min[a, count], combine_max[a, count] = (
                            IDENTITY_MAP_FACTOR,
                            IDENTITY_MAP_FACTOR,
                        )
                        count += 1
                    elif a < 2**num_blocks - n and b >= 2**num_blocks - n:
                        split_a[count, b], split_b[count, b] = 1, 1
                        combine_min[b, count], combine_max[b, count] = (
                            IDENTITY_MAP_FACTOR,
                            IDENTITY_MAP_FACTOR,
                        )
                        count += 1
                    else:
                        assert False

            split_a = split_a[:count, 2**num_blocks - n :]
            split_b = split_b[:count, 2**num_blocks - n :]
            combine_min = combine_min[2**num_blocks - n :, :count]
            combine_max = combine_max[2**num_blocks - n :, :count]
            network.append((split_a, split_b, combine_min, combine_max))

    return network


def odd_even_network(n):
    layers = n

    network = []

    shifted: bool = False
    even: bool = n % 2 == 0

    for _ in range(layers):

        if even:
            k = n // 2 + shifted
        else:
            k = n // 2 + 1

        split_a, split_b = np.zeros((k, n)), np.zeros((k, n))
        combine_min, combine_max = np.zeros((n, k)), np.zeros((n, k))

        count = 0

        # for i in range(n // 2 if not (even and shifted) else n // 2 - 1):
        for i in range(int(shifted), n - 1, 2):
            a, b = i, i + 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = 1, 1
            count += 1

        if even and shifted:
            # Make sure that the corner values stay where they are/were:
            a, b = 0, 0
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = 0.5, 0.5
            count += 1
            a, b = n - 1, n - 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = 0.5, 0.5
            count += 1

        elif not even:
            if shifted:
                a, b = 0, 0
            else:
                a, b = n - 1, n - 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = 0.5, 0.5
            count += 1

        assert count == k

        network.append((split_a, split_b, combine_min, combine_max))
        shifted = not shifted

    return network


def s_best(x):
    return (
        torch.clamp(x, -0.25, 0.25)
        + 0.5
        + ((x > 0.25).float() - (x < -0.25).float())
        * (0.25 - 1 / 16 / (x.abs() + 1e-10))
    )


def execute_sort(
    sorting_network, vectors, steepness=10.0, art_lambda=0.25, distribution="cauchy"
):
    x = vectors
    X = torch.eye(vectors.shape[1], dtype=x.dtype, device=x.device).repeat(
        x.shape[0], 1, 1
    )

    for split_a, split_b, combine_min, combine_max in sorting_network:
        split_a = split_a.type(x.dtype)
        split_b = split_b.type(x.dtype)
        combine_min = combine_min.type(x.dtype)
        combine_max = combine_max.type(x.dtype)

        a, b = x @ split_a.T, x @ split_b.T

        # float conversion necessary as PyTorch doesn't support Half for sigmoid as of 25. August 2021
        new_type = torch.float32 if x.dtype == torch.float16 else x.dtype

        if distribution == "logistic":
            alpha = torch.sigmoid((b - a).type(new_type) * steepness).type(x.dtype)

        elif distribution == "logistic_phi":
            alpha = torch.sigmoid(
                (b - a).type(new_type)
                * steepness
                / ((a - b).type(new_type).abs() + 1.0e-10).pow(art_lambda)
            ).type(x.dtype)

        elif distribution == "gaussian":
            v = (b - a).type(new_type)
            alpha = NormalCDF.apply(v, 1 / steepness)
            alpha = alpha.type(x.dtype)

        elif distribution == "reciprocal":
            v = steepness * (b - a).type(new_type)
            alpha = 0.5 * (v / (2 + v.abs()) + 1)
            alpha = alpha.type(x.dtype)

        elif distribution == "cauchy":
            v = steepness * (b - a).type(new_type)
            alpha = 1 / math.pi * torch.atan(v) + 0.5
            alpha = alpha.type(x.dtype)

        elif distribution == "optimal":
            v = steepness * (b - a).type(new_type)
            alpha = s_best(v)
            alpha = alpha.type(x.dtype)

        else:
            raise NotImplementedError(
                "softmax method `{}` unknown".format(distribution)
            )

        aX = X @ split_a.T
        bX = X @ split_b.T
        w_min = alpha.unsqueeze(-2) * aX + (1 - alpha).unsqueeze(-2) * bX
        w_max = (1 - alpha).unsqueeze(-2) * aX + alpha.unsqueeze(-2) * bX
        X = (w_max @ combine_max.T.unsqueeze(-3)) + (
            w_min @ combine_min.T.unsqueeze(-3)
        )
        x = (alpha * a + (1 - alpha) * b) @ combine_min.T + (
            (1 - alpha) * a + alpha * b
        ) @ combine_max.T
    return x, X


class NormalCDF(torch.autograd.Function):
    def forward(ctx, x, sigma):
        ctx.save_for_backward(x, torch.tensor(sigma))
        return 0.5 + 0.5 * torch.erf(x / sigma / math.sqrt(2))

    def backward(ctx, grad_y):
        x, sigma = ctx.saved_tensors
        return (
            grad_y
            * 1
            / sigma
            / math.sqrt(math.pi * 2)
            * torch.exp(-0.5 * (x / sigma).pow(2)),
            None,
        )


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


class SurvivalModel(nn.Module):
    def __init__(
        self, embedding_dim, num_structured_features, sorter_size=32, num_documents=5
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_documents = num_documents
        self.attention_aggregator = AttentionAggregator(embedding_dim, num_documents)
        self.fc = nn.Linear(embedding_dim + num_structured_features, sorter_size)
        self.sorter = CustomDiffSortNet(
            sorting_network_type="bitonic",
            size=sorter_size,
            steepness=2 * sorter_size,
            distribution="cauchy",
        )

    def forward(self, document_embeddings, time_diffs, structured_data):
        risk_scores = self.get_risk_scores(
            document_embeddings, time_diffs, structured_data
        )
        _, perm_prediction = self.sorter(risk_scores)
        return risk_scores, perm_prediction

    def get_risk_scores(self, document_embeddings, time_diffs, structured_data):
        document_embeddings = F.normalize(document_embeddings, p=2, dim=2)
        aggregated_embedding = self.attention_aggregator(
            document_embeddings, time_diffs
        )
        combined_features = torch.cat([aggregated_embedding, structured_data], dim=1)
        raw_risk_score = self.fc(combined_features)
        return raw_risk_score
