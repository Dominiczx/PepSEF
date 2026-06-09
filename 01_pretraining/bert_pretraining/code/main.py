import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from models.bert_model import BERTModel, initialize_weights
from utils.tokenizer import PeptideTokenizer
from utils.dataset import PeptideDataset, process_fasta_file, collate_fn
from transformers import get_scheduler

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train_model(model, dataloader, optimizer, device, epochs, save_path):
    criterion = torch.nn.CrossEntropyLoss()
    num_training_steps = len(dataloader) * epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    min_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device).long()
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            mlm_positions = batch['mlm_positions'].to(device).long()

            optimizer.zero_grad()
            _, logits = model(input_ids, attention_mask, mlm_positions)

            # Align logits and labels for masked positions
            logits = logits.view(-1, logits.size(-1))  # Shape: (num_masked_positions, vocab_size)
            masked_labels = labels[torch.arange(labels.size(0)).unsqueeze(1), mlm_positions]  # Align labels with mlm_positions
            masked_labels = masked_labels.view(-1)  # Flatten the labels

            if logits.size(0) != masked_labels.size(0):
                raise ValueError(f"Logits and masked_labels size mismatch: {logits.size(0)} vs {masked_labels.size(0)}")

            # Compute loss
            loss = criterion(logits, masked_labels)
            total_loss += loss.item()

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)  # Get predicted token IDs
            correct_predictions += (predictions == masked_labels).sum().item()
            total_predictions += masked_labels.size(0)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    # Load configuration
    config = yaml.safe_load(open(args.config, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer and dataset
    tokenizer = PeptideTokenizer(config['tokenizer'])
    sequences = process_fasta_file(config['dataset']['path'], config['tokenizer']['max_length'], tokenizer)
    print(len(sequences), "sequences loaded from dataset")
    exit(0)
    dataset = PeptideDataset(sequences, tokenizer, config['tokenizer']['max_length'])
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)

    # Initialize model
    bert_model = BERTModel(tokenizer.vocab_size, **config['model']['bert']).to(device)
    # optimizer = torch.optim.AdamW(bert_model.parameters(), lr=float(config['training']['learning_rate']))
    optimizer = torch.nn.BCELoss()
    bert_model.apply(initialize_weights)
    # Train model
    train_model(bert_model, dataloader, optimizer, device, config['training']['epochs'], config['save_path'])

if __name__ == '__main__':
    main()