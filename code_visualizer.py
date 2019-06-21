import torch
from radio_transformer_networks import Net, CHANNEL_USE, BLOCK_SIZE

snr = 7
NUM_TRAINING = 1
model = Net(2**BLOCK_SIZE, compressed_dim=CHANNEL_USE, snr=snr)
model.load_state_dict(torch.load('./models/model_with_fancy_mod'+str(snr)))

train_labels = (torch.rand(NUM_TRAINING) * (2**BLOCK_SIZE)).long()
train_data = torch.eye(2**BLOCK_SIZE).index_select(dim=0, index=train_labels)

print(train_data)
print(model.encode(train_data))
