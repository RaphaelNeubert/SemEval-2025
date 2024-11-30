
import random
import re
from itertools import chain
from collections import Counter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from dataclasses import dataclass

@dataclass
class DataConfig:
    train_path: str = "data/merged_dataset_train.csv"
    valid_path: str = "data/merged_dataset_valid.csv"
    test_path: str =  "data/merged_dataset_test.csv"
    vocab_size: int = 30000
    batch_size_train: int = 64
    batch_size_eval: int = 64
    batch_size_test: int = 64
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"


class Vocabulary:
    # <PAD> token always gets index 0
    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.special_tokens = [pad_token, unk_token]
        self.word_to_index = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.index_to_word = list(self.word_to_index.keys())

    def build(self, data: list[list[str]], vocab_size: int):
        words = chain.from_iterable(data)
        counter = Counter(words).most_common(vocab_size-2)

        self.word_to_index.update({word: idx + len(self.special_tokens) for idx, (word, _) in enumerate(counter)})
        self.index_to_word = list(self.word_to_index.keys())

    def words_to_indices(self, words: list[str]) -> list[int]:
        return [self.word_to_index.get(word, self.word_to_index.get(self.unk_token)) for word in words]

    def index_to_words(self, indices: list[int]) -> list[str]:
        return [self.index_to_word[idx] for idx in indices]

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                "word_to_index": self.word_to_index,
                "index_to_word": self.index_to_word
            }, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        
        vocab = cls()
        vocab.word_to_index = data["word_to_index"]
        vocab.index_to_word = data["index_to_word"]
        return vocab

    def size(self):
        return len(self.index_to_word)

    def print_batch(self, batch: torch.Tensor):
        for sample in batch:
            indices = sample[~(sample==0)]  # remove padding
            print(self.index_to_words(indices))

def load_data(datapath: str) -> list[list[str, list[int]]]:
    with open(datapath) as f:
        f.readline() # ignore first line 
        data = f.readlines()
    data = [[l.split('\t')[1], list(map(int, l[:-1].split('\t')[2:]))] for l in data]
    return data


def process_text(data: list[list[str, list[int]]]) -> list[list[list[str], list[int]]]:
    for d in data:
        d[0] = d[0].lower()
        d[0] = re.sub(r'([.,!?()"\'])', r' \1 ', d[0]) # add whitespace around punctuation
        d[0] = d[0].split()
    return data

def prep_batch(batch):
    inputs = [torch.tensor(x[0]) for x in batch]
    targets = torch.tensor([x[1] for x in batch])
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)  # corresponds to <PAD>
    mask = ~(inputs_padded == 0)
    return inputs_padded, mask, targets

def get_data(config: DataConfig) -> (DataLoader, DataLoader, DataLoader, Vocabulary): 
    train_data = load_data(config.train_path)
    eval_data = load_data(config.valid_path)
    test_data = load_data(config.test_path)

    train_tokens = process_text(train_data)
    eval_tokens = process_text(eval_data)
    test_tokens = process_text(test_data)

    vocab = Vocabulary(pad_token=config.pad_token, unk_token=config.unk_token)
    text = [t[0] for t in train_tokens+eval_tokens]
    vocab.build(text, vocab_size=config.vocab_size)

    train_indices = [[vocab.words_to_indices(d[0]), d[1]] for d in train_tokens]
    eval_indices = [[vocab.words_to_indices(d[0]), d[1]] for d in eval_tokens]
    test_indices = [[vocab.words_to_indices(d[0]), d[1]] for d in eval_tokens]

    trainloader = DataLoader(train_indices, batch_size=config.batch_size_train, collate_fn=prep_batch, shuffle=True)
    evalloader = DataLoader(eval_indices, batch_size=config.batch_size_eval, collate_fn=prep_batch, shuffle=True)
    testloader = DataLoader(test_indices, batch_size=config.batch_size_test, collate_fn=prep_batch, shuffle=True)

    return trainloader, evalloader, testloader, vocab

