import torch

def generate_data(block_size, use_complex):
    train_data = torch.randint(2, (10240, block_size)).float()
    test_data = torch.randint(2, (2500, block_size)).float()
    if use_complex:
        train_zeros = torch.zeros(10240, block_size*2).float()
        test_zeros = torch.zeros(2500, block_size*2).float()
        train_zeros[:, :block_size] = train_data
        test_zeros[:, :block_size] = test_data
        train_data = train_zeros
        test_data = test_zeros

    train_labels = train_data
    test_labels = test_data
    return train_data, train_labels, test_data, test_labels

