import re


def getTokens(file:str):
  
  with open(file,'r',encoding='utf-8') as f:
    raw_text = f.read()

  
    
    raw_text = re.split(r'([,,:;?_!"()\']|--|\s)',raw_text)

    raw_text = [item.strip() for item in raw_text if item.strip()]
    
    return raw_text


def tokenization(tokens:list[str]):
  all_words = sorted(set(tokens))
  #size = len(all_words)
  words_dict = {token:id for id,token in enumerate(all_words)}

  return words_dict

tokens = getTokens('test.txt')

vocab = tokenization(tokens)

print(vocab)
