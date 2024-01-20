import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset

from transformers import TextDataset, DataCollatorForLanguageModeling

import logging
logging.basicConfig(level=logging.INFO)

import random

def select_top_k(predictions, k=10):
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return predicted_index
train_model = "./checkpoint-10000"
# 载入预训练模型的分词器
tokenizer = AutoTokenizer.from_pretrained('./gpt2-medium-tokenizer')
# 读取 GPT-2 预训练模型
model = GPT2LMHeadModel.from_pretrained(train_model)
model.eval()

text = " "
indexed_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([indexed_tokens])
tokens_tensor.shape

total_predicted_text = text

def check(text):
    n = len(text)
    i = 0
    num = 0
    while i < n:
        if text[i] == '.' or text[i] == '!' or text[i] == '?':
            while i < n-1 and text[i+1] == text[i]:
                i += 1
            num += 1
            continue
        if num >= 5:
            return False
    return True

for _ in range(200):
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    predicted_index = select_top_k(predictions, k=10)
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    total_predicted_text += tokenizer.decode(predicted_index)

    if '<|endoftext|>' in total_predicted_text or not check(total_predicted_text):
        break

    indexed_tokens += [predicted_index]
    tokens_tensor = torch.tensor([indexed_tokens])

print(total_predicted_text)
