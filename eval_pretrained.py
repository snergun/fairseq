from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.data.dictionary import Dictionary
from fairseq_cli.eval_lm import eval_lm
import torch
import json
from datasets import load_dataset
import re

def load_and_tokenize(vocab_path, split, unk_id = 3):
    with open(vocab_path, 'r') as file:
        vocab = json.load(file)
    raw_datasets = load_dataset("wikitext", "wikitext-103-raw-v1")
    SPACE_NORMALIZER = re.compile(r"\s+")
    tokens = ' '.join(SPACE_NORMALIZER.sub(" ", line).strip('\n').strip() for line in raw_datasets[split]['text']).split()
    token_ids = [vocab.get(token,unk_id) for token in tokens]
    token_ids = torch.Tensor(token_ids).long()
    return token_ids

class DataLoaderLite:

    def __init__(self, tokens, B, T, process_rank = 0, num_processes = 1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.tokens = tokens
        self.reset()
        self.id = -1
    def reset(self):
        # state, init at shard zero
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        self.id += 1
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset position
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.reset()
        return {"net_input" : {"src_tokens" : x}, "target" : y, "ntokens" : y.size(0) * y.size(1), "id": [self.id]}
    
    def __iter__(self):
        for step in range(len(self)):
            yield self.next_batch()
            
    def __len__(self):
        return len(self.tokens) // (self.B * self.T * self.num_processes)

device = "cuda:" + str(0) if torch.cuda.is_available() else "cpu"
    
fairseq_model = TransformerLanguageModel.from_pretrained('/base-vol-2/fairseq/models/adaptive_lm_wiki103.v2/', 'model.pt')
fairseq_model.to(device)
source_dictionary = Dictionary.load('/base-vol-2/fairseq/models/adaptive_lm_wiki103.v2/dict.txt')

#Load the Datasets
vocab_path = '/base-vol-2/fairseq/models/adaptive_lm_wiki103.v2/vocab.txt'
# train_data = load_and_tokenize(vocab_path, 'train')
val_data = load_and_tokenize(vocab_path, 'test')


#Create data samples of length = block_size and data collator
B = 1 # micro batch size
T = 512    
# train_data_loader = DataLoaderLite(torch.tensor(train_data), B, T)
val_data_loader = DataLoaderLite(val_data, B, T)

print("Dataloaders instantiated")


results = eval_lm(fairseq_model.models, source_dictionary, val_data_loader,device=device)
print(results)