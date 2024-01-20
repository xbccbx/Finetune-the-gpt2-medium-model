import torch
import math
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from datasets import load_dataset
from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained('./gpt2-medium-tokenizer')

def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True, max_length = 512, padding='max_length')

tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# 加载和预处理数据集
dataset = load_dataset('csv', data_files='processed_dataset.csv', column_names=['text'])
tokenized_dataset = dataset.map(preprocess_function, batched=True)
t_dataset, v_dataset = random_split(tokenized_dataset['train'], lengths=[0.9, 0.1], random_state = 42)

model = GPT2LMHeadModel.from_pretrained("./gpt2-medium-model").to(device)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir = "./output_dir",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True,
    use_cpu=False,
)
print(training_args.device)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

trainer.train()
eval_results = trainer.evaluate(v_dataset)
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")