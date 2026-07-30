import numpy as np
from torch.utils.data import Subset
from typing import List
from .base_dataset import BaseFLDataset


def partition_dataset(
    dataset: BaseFLDataset,
    num_clients: int,
    iid: bool = False,  # 默认 Non-IID
    seed: int = 42,
    alpha: float = 0.2,
    by_writer: bool = False
) -> List[Subset]:
    """
    Partitions a given dataset object among multiple clients.

    Args:
        dataset (BaseFLDataset): 数据集对象.
        num_clients (int): 客户端数量.
        iid (bool): True 表示 IID，False 表示按标签 Non-IID.
        seed (int): 随机种子.
        alpha (float): Dirichlet 分布的浓度参数.
        by_writer (bool): True 表示按数据集自带的 writer 划分分区
            （仅适用于 FedMnistDataset 等支持 client_indices 的数据集）.

    Returns:
        List[Subset]: 按客户端划分好的 Subset 列表.
    """
    if by_writer:
        partitions = _partition_by_writer(dataset, num_clients, seed)
    elif iid:
        partitions = _partition_iid(dataset, num_clients, seed)
    else:
        partitions = _partition_non_iid(
            dataset, num_clients, seed, alpha
        )

    print(f"Partitioned dataset into {len(partitions)} "
          f"{'by-writer' if by_writer else ('IID' if iid else 'Non-IID')} subsets.")
    return partitions


def _partition_by_writer(
    dataset: BaseFLDataset,
    num_clients: int,
    seed: int
) -> List[Subset]:
    """
    按数据集自带的 writer（写作者）划分分区。

    适用于 FedMnistDataset 等具有 client_indices 属性的数据集。
    当 writer 数量多于 num_clients 时，将多个 writer 合并到一个客户端；
    当 writer 数量少于 num_clients 时，报错提示。

    Args:
        dataset (BaseFLDataset): 具有 client_indices 属性的数据集.
        num_clients (int): 客户端数量.
        seed (int): 随机种子.

    Returns:
        List[Subset]: 按客户端划分好的 Subset 列表.
    """
    if not hasattr(dataset, 'client_indices'):
        raise ValueError(
            "Dataset does not support by_writer partitioning. "
            "It must have a 'client_indices' property (e.g., FedMnistDataset)."
        )

    writer_indices = dataset.client_indices
    num_writers = len(writer_indices)

    if num_writers < num_clients:
        raise ValueError(
            f"Number of writers ({num_writers}) is less than num_clients ({num_clients}). "
            f"Cannot partition by writer with more clients than writers."
        )

    # Shuffle writer order for randomness
    rng = np.random.default_rng(seed)
    writer_order = list(range(num_writers))
    rng.shuffle(writer_order)

    # Distribute writers to clients
    # Each client gets at least num_writers // num_clients writers,
    # and the first (num_writers % num_clients) clients get one extra writer
    writers_per_client = [num_writers // num_clients] * num_clients
    for i in range(num_writers % num_clients):
        writers_per_client[i] += 1

    client_subsets = []
    writer_offset = 0
    for i in range(num_clients):
        # Get the writer indices assigned to this client
        assigned_writers = writer_order[writer_offset:writer_offset + writers_per_client[i]]
        writer_offset += writers_per_client[i]

        # Merge all sample indices from assigned writers
        sample_indices = []
        for w in assigned_writers:
            sample_indices.extend(writer_indices[w])

        # Shuffle sample indices within each client
        rng.shuffle(sample_indices)
        client_subsets.append(Subset(dataset, sample_indices))

    return client_subsets


def _partition_iid(dataset: BaseFLDataset, num_clients: int, seed: int) -> List[Subset]:
    """IID 分区"""
    num_items_per_client = len(dataset) // num_clients
    all_indices = list(range(len(dataset)))

    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)

    client_indices = [
        all_indices[i * num_items_per_client:(i + 1) * num_items_per_client]
        for i in range(num_clients)
    ]
    return [Subset(dataset, indices) for indices in client_indices]


# def _partition_non_iid_by_label(
#     dataset: BaseFLDataset,
#     num_clients: int,
#     seed: int,
#     shards_per_client: int
# ) -> List[Subset]:
#     """按标签 Non-IID 分区"""
#     num_shards = num_clients * shards_per_client
#     if len(dataset) < num_shards:
#         raise ValueError("Not enough data for the specified number of shards.")
#     num_images_per_shard = len(dataset) // num_shards

#     labels = np.array(dataset.targets)
#     indices_by_label = np.argsort(labels)

#     shards = [
#         indices_by_label[i * num_images_per_shard:(i + 1) * num_images_per_shard]
#         for i in range(num_shards)
#     ]

#     rng = np.random.default_rng(seed)
#     rng.shuffle(shards)

#     client_indices = [
#         np.concatenate(shards[i * shards_per_client:(i + 1) * shards_per_client]).tolist()
#         for i in range(num_clients)
#     ]
#     return [Subset(dataset, indices) for indices in client_indices]

def _partition_non_iid(
    dataset: BaseFLDataset,
    num_clients: int,
    seed: int,
    alpha: float
) -> List[Subset]:
    """
    Partitions a dataset into Non-IID subsets for a number of clients
    using a Dirichlet distribution.
    This method simulates a real-world scenario where the data distribution
    across clients is skewed, controlled by the `alpha` parameter.
    Args:
        dataset (BaseFLDataset): The dataset to partition. It must have a
            `targets` property that returns a list of labels.
        num_clients (int): The number of clients to partition the data for.
        seed (int): The random seed for reproducibility of the partition.
        alpha (float): The concentration parameter for the Dirichlet
            distribution. A smaller alpha (e.g., 0.1) creates a more
            skewed (Non-IID) distribution, while a larger alpha (e.g., 100)
            results in a more uniform (IID-like) distribution.
    Returns:
        List[Subset]: A list of PyTorch Subset objects, where each element
            corresponds to a client's local dataset.
    """
    # --- Step 1: Setup and Initialization ---
    
    # Set the random seed for NumPy for reproducible partitioning
    np.random.seed(seed)
    # Get the total number of samples and the list of labels
    labels = np.array(dataset.targets)
    num_samples = len(labels)
    
    # Get the number of unique classes in the dataset
    num_classes = len(np.unique(labels))
    # --- Step 2: Group Data Indices by Class ---
    
    # Create a list of lists, where class_indices[i] contains the
    # indices of all samples belonging to class i.
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    # --- Step 3: Partition Data using Dirichlet Distribution ---
    # This list will hold the data indices for each client.
    # client_indices[i] will be the list of indices for client i.
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    # For each class, distribute its samples among the clients
    for k_indices in class_indices:
        # Shuffle the indices of the current class to ensure random assignment
        np.random.shuffle(k_indices)
        # Sample proportions from a Dirichlet distribution.
        # This vector of proportions determines how the samples of the current
        # class are split among the clients.
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Ensure that the distribution is not too skewed for very small classes,
        # which can cause issues. We adjust proportions to prevent clients
        # from getting fractions of a sample.
        # The sum of proportions should be balanced with the number of samples.
        proportions = np.array([p * (len(idx_j) < num_samples / num_clients) for p, idx_j in zip(proportions, client_indices)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(k_indices)).astype(int)[:-1]
        
        # Use np.split to divide the class indices based on the calculated proportions.
        # This is an efficient way to create the splits.
        # `split_k_indices` will be a list of arrays, one for each client.
        split_k_indices = np.split(k_indices, proportions)
        
        # Assign the split indices to each client
        for i in range(num_clients):
            client_indices[i].extend(split_k_indices[i].tolist())
    # --- Step 4: Create PyTorch Subset Objects ---
    # Create a list of Subset objects from the partitioned indices
    client_subsets = []
    for indices in client_indices:
        # Ensure indices are shuffled for each client's local training
        print(len(indices))
        np.random.shuffle(indices)
        client_subsets.append(Subset(dataset, indices))
    return client_subsets
