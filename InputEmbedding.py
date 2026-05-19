import torch
from Tokenization import GPTDatasetV1, create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()

# Example usage
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
out_dim = 3
torch.manual_seed(123)
emb_layer = torch.nn.Embedding(vocab_size, out_dim)
print(emb_layer.weight)
print(emb_layer(torch.tensor(torch.tensor(input_ids))))


#creating input embeddings for gpt2 for a batch of input sequences

vocab_size = 50257
output_dim = 256
token_emb_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, window_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print(inputs.shape)
# print(token_emb_layer(inputs).shape)
token_emb = token_emb_layer(inputs)
print(token_emb.shape)
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
input_emb = token_emb + pos_embeddings
print(input_emb.shape)