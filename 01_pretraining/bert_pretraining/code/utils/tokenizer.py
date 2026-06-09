import json
import re

class PeptideTokenizer():
    def __init__(self, tokenizer_config) -> None:
        self.vocab = {}
        self.init_vocab(tokenizer_config['vocab_path'], tokenizer_config['special_token_path'])
        self.max_length = tokenizer_config['max_length']
        self.vocab_size = len(self.vocab)
        
    def init_vocab(self, vocab_path, special_token_path):
        with open(special_token_path, "r", encoding='utf-8') as f:
            self.special_token = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            for index, token in enumerate(f):
                token = token.strip()
                self.vocab[token] = index
            self.inv_vocab = {v: k for k,v in self.vocab.items()}
        
    def token_to_id(self, one):
        one = re.sub('[UZOBuzob]', 'x', one.lower())
        if one == 'x':
            return self.vocab[self.special_token['unk_token']]
        elif one in self.vocab:
            return self.vocab[one]
        else:
            return self.vocab[self.special_token['pad_token']]
        
    def encode(self, seq, padding = False):
        outputs = []
        seq = re.sub('[UZOBuzob]', 'x', seq.lower())
        outputs.append(self.vocab[self.special_token['cls_token']])
        for i, s in enumerate(seq):
            if i >= self.max_length - 2: break
            if s == "x":
                outputs.append(self.vocab[self.special_token['unk_token']])
            else:
                outputs.append(self.vocab[s])
        outputs.append(self.vocab[self.special_token['sep_token']])
        if len(outputs) < self.max_length and padding:
            outputs.extend([self.vocab[self.special_token['pad_token']] for _ in range(self.max_length - len(outputs))])
        return outputs
    
    def convert_ids_to_tokens(self, ids):
        outputs = []
        for i in ids:
            outputs.append(self.inv_vocab[i])
        return outputs
    
    def encode_plus(self, seq, padding=False):
        outputs = {}
        outputs['input_ids'] = self.encode(seq, padding)
        outputs['attention_mask'] = [1 if i != self.vocab[self.special_token['pad_token']] else 0 for i in outputs['input_ids']]
        return outputs
        # return outputs['input_ids'], outputs['attention_mask']
    
    def decode(self, ids):
        return " ".join(self.convert_ids_to_tokens(ids))