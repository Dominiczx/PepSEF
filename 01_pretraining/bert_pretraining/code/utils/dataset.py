import re
import random
from torch.utils.data import Dataset
import torch

def process_fasta_file(fasta_path, max_length, tokenizer):
    sequences = []
    with open(fasta_path, "r") as f:
        sequence = ""
        for line in f:
            if line.startswith(">"):
                if sequence:
                    sequences.append(sequence)
                sequence = ""
            else:
                sequence += re.sub(r"\s+", "", line.strip())
        if sequence:
            sequences.append(sequence)

    processed_sequences = []
    for seq in sequences:
        if len(seq) > max_length:
            seq = seq[:max_length - 2]  # Reserve space for [CLS] and [SEP]
        processed_sequences.append(seq)

    return processed_sequences

def mask_seq(src, tokenizer, mask_prob=0.15):
    """
    Masks a sequence for MLM training.
    Args:
        src: List of token IDs (input sequence).
        tokenizer: Tokenizer object with vocab and special tokens.
        mask_prob: Probability of masking a token.
    Returns:
        masked_src: Masked input sequence.
        labels: Original token IDs for masked positions (0 for non-masked).
    """
    MASK_TOKEN = tokenizer.vocab[tokenizer.special_token['mask_token']]
    PAD_TOKEN = tokenizer.vocab[tokenizer.special_token['pad_token']]
    CLS_TOKEN = tokenizer.vocab[tokenizer.special_token['cls_token']]
    SEP_TOKEN = tokenizer.vocab[tokenizer.special_token['sep_token']]

    masked_src = src[:]
    labels = [0] * len(src)

    for i in range(len(src)):
        if src[i] in [PAD_TOKEN, CLS_TOKEN, SEP_TOKEN]:
            continue

        if random.random() < mask_prob:
            labels[i] = src[i]  # Save the original token
            prob = random.random()

            if prob < 0.8:
                # Replace with [MASK]
                masked_src[i] = MASK_TOKEN
            elif prob < 0.9:
                # Replace with a random token
                masked_src[i] = random.randint(1, len(tokenizer.vocab) - 1)
            else:
                # Keep the original token (10% chance)
                masked_src[i] = src[i]

    return masked_src, labels

def collate_fn(batch):
    max_len = max(len(item['input_ids']) for item in batch)
    max_mlm_positions = max(len(item['mlm_positions']) for item in batch)

    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        item['input_ids'] = torch.cat([item['input_ids'], torch.tensor([0] * pad_len, dtype=torch.long)])
        item['attention_mask'] = torch.cat([item['attention_mask'], torch.tensor([0] * pad_len, dtype=torch.long)])
        item['labels'] = torch.cat([item['labels'], torch.tensor([0] * pad_len, dtype=torch.long)])
        item['mlm_positions'] = torch.cat([item['mlm_positions'], torch.tensor([0] * (max_mlm_positions - len(item['mlm_positions'])), dtype=torch.long)])

    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'mlm_positions': torch.stack([item['mlm_positions'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }

class PeptideDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoding = self.tokenizer.encode_plus(seq, padding=True)
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Ensure padding to max_length
        input_ids = input_ids + [self.tokenizer.vocab[self.tokenizer.special_token['pad_token']]] * (self.max_length - len(input_ids))
        attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))
        
        # Apply masking
        masked_input_ids, labels = mask_seq(input_ids, self.tokenizer)

        return {
            'input_ids': torch.tensor(masked_input_ids[:self.max_length]),
            'attention_mask': torch.tensor(attention_mask[:self.max_length]),
            'mlm_positions': torch.tensor([i for i, label in enumerate(labels) if label != 0]),
            'labels': torch.tensor(labels[:self.max_length])
        }