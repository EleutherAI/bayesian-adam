import json

import torch
from torch.utils.data import Dataset, DataLoader
from x_transformers import TransformerWrapper, Decoder
from adasghmc import AdaSGHMC 

from torch.profiler import profile, record_function, ProfilerActivity

class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.data = data
        self.chars = sorted(list(set(self.data)))
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        input_seq = torch.tensor([self.char_to_int[char] for char in chunk[:-1]], dtype=torch.long)
        target_seq = torch.tensor([self.char_to_int[char] for char in chunk[1:]], dtype=torch.long)
        return input_seq, target_seq

def load_char_dataset(filepath, seq_len):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    return CharDataset(text, seq_len)

def create_model(seq_len, num_chars, dim, depth, heads):
    model = TransformerWrapper(
        num_tokens=num_chars,
        max_seq_len=seq_len,
        attn_layers=Decoder(
            dim=dim,
            depth=depth,
            heads=heads
        )
    #).cuda()
    )
    return model

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config


def train_model(model, dataloader, num_epochs=10):
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), 
    #            lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    optimizer = AdaSGHMC(model.parameters(), 
                learning_rate=1e-3, gradient_ema=0.999, 
                momentum=0.9, eps=1e-8)

    model.train()

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            #inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            outputs = outputs.view(-1, model.num_tokens)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)*100000
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


def main(config_path):
    config = load_config(config_path)
    print("A")

    dataset = load_char_dataset(config["filepath"], config["seq_len"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    print("B")

    model = create_model(config["seq_len"], 
            len(dataset.chars), config["model"]["dim"], 
            config["model"]["depth"], config["model"]["heads"])
    print("C")
    train_model(model, dataloader, config["num_epochs"])

if __name__ == "__main__":
    config_path = 'config.json'  # Path to your config file
    main(config_path)
