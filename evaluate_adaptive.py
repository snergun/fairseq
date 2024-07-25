from fairseq.models.transformer_lm import TransformerLanguageModel
from transformers import TransfoXLTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch 
import numpy as np
import os

os.environ["TRUST_REMOTE_CODE"] = "True"
local_rank = 0
device = "cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)
device_type = "cuda" if device.startswith("cuda") else "cpu"

#Load Pretrained Model
model = TransformerLanguageModel.from_pretrained('/base-vol-2/fairseq/models/adaptive_lm_wiki103.v2/', 'model.pt')
model = model.models[0]
model.to(device)

#Load Dataset 
tokenized_datasets = load_from_disk('/base-vol-2/datasets/wikitext_tokenized_txl')
print("Dataset Loaded")

checkpoint = 'transfo-xl/transfo-xl-wt103'
revision = '40a186da79458c9f9de846edfaea79c412137f97'
tokenizer = TransfoXLTokenizer.from_pretrained(checkpoint, revision=revision)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer Loaded")

def group_texts(examples):
    total_length = len(examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if block_size == None:
        return examples
    else:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in examples.items()
        }
    return result

#Create data samples of length = block_size and data collator
block_size = 512 
tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

B = 32 # micro batch size

class CustomBatchSampler():
    def __init__(self,data,batch_size,n_proc):
        self.data = data
        self.data_len = len(data)
        self.batch_size = batch_size
        self.n_proc = n_proc
        self.start_ids = np.arange(batch_size) * len(data) / n_proc / batch_size
        self.len = self.data_len//(self.batch_size*self.n_proc) * self.n_proc

    def __len__(self):
        return int(self.len)

    def __iter__(self):
        for step in range(self.len):
            n_proc = self.n_proc
            yield list( ( ( self.start_ids + (self.data_len / n_proc) * ( step % n_proc) ) + (step // n_proc ) ).astype(int) )

# Instantiate dataloaders.
test_batch_sampler = CustomBatchSampler(data = tokenized_datasets['test'], batch_size = B, n_proc = 1)
test_dataloader = DataLoader(tokenized_datasets['test'], batch_sampler = test_batch_sampler , collate_fn=data_collator)
print("Dataloaders instantiated")

model.eval()
eval_losses = torch.zeros(len(test_dataloader), device=device)

for eval_step, batch in enumerate(test_dataloader):
    with torch.no_grad():
        x = batch['input_ids'].to(device)
        y = batch['labels'].to(device)
        outputs = model(x, y)
    eval_losses[eval_step] = outputs.loss

eval_metric = eval_losses.mean()
print(eval_metric)