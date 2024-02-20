from transformers import EsmTokenizer, EsmModel
import torch

tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
seqs =["QERLKSIVRILE", "QERLKSIVRILEEEERRRRRRFFFFFRRRFFRRFRRFFRFFR"]
inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
x = last_hidden_states.detach()
print(x.mean(axis=1))
