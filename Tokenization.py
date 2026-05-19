import re

def getTokens(file:str):
  with open(file,'r',encoding='utf-8') as f:
    raw_text = f.read()
    raw_text = re.split(r'([,,:;?_!"()\']|--|\s)',raw_text)
    raw_text = [item.strip() for item in raw_text if item.strip()]
    return raw_text
  
def CreateVocab(tokens:list[str]):
  all_words = sorted(set(tokens))
  all_words.extend(["<|endoftext|>","<|unknown|>"])
  #size = len(all_words)
  words_dict = {token:id for id,token in enumerate(all_words)}
  return words_dict

class Tokenizer:
  def __init__(self, vocab:dict[str,int]):
    self.vocab = vocab
    self.inv_vocab = {id:token for token,id in vocab.items()}

  def encode(self, text:str):
    tokens = re.split(r'([,,:;?_!"()\']|--|\s)',text)
    tokens = [item.strip() if item.strip() in self.vocab else "<|unknown|>" for item in tokens if item.strip()]
    return [self.vocab[token] for token in tokens]

  def decode(self, ids:list[int]):
    decoded_text =" ".join([self.inv_vocab[id] for id in ids])
    decoded_text = re.sub(r'\s+([,.?!"()\'])', r'\1', decoded_text)
    return decoded_text

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = ("Hello! How are you alenziiqnden ier <|endoftext|> This is a test of the GPT-2 tokenizer.")
ints = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
strings = tokenizer.decode(ints)
# print(ints)
# print(strings)

# -------------
# Data Sampling
# -------------
with open("the-verdict.txt", "r", encoding="utf-8") as f:
  raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]
context_size = 4
for i in range(1, context_size+1):
  context = enc_sample[:i]
  target = enc_sample[i]
  print(tokenizer.decode(context), "---->", tokenizer.decode([target]))

import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
  def __init__(self, txt, tokenizer, window_length, stride):
    self.input_ids = []
    self.target_ids = []

    token_ids = tokenizer.encode(txt)

    for i in range(0, len(token_ids) - window_length, stride):
      input_chunk = token_ids[i:i + window_length]
      target_chunk = token_ids[i + 1: i + window_length + 1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))
    
  def __len__(self):
   return len(self.input_ids)
  
  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx]
  
def create_dataloader_v1(txt, batch_size=4, window_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
  tokenizer = tiktoken.get_encoding("gpt2")
  dataset = GPTDatasetV1(txt, tokenizer, window_length, stride)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
  )
  return dataloader

dataloader = create_dataloader_v1(raw_text, batch_size=1, window_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)

dataloader = create_dataloader_v1(raw_text, batch_size=8, window_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


