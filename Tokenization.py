import re

import tiktoken
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




tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello! How are you alenziiqnden ier <|endoftext|> This is a test of the GPT-2 tokenizer.")

ints = tokenizer.encode(text, allowed_special={"<|endoftext|>"})


strings = tokenizer.decode(ints)


print(ints)
print(strings)