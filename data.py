from collections import Counter
from itertools import chain
import torch
import re
import h5py
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"

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
        counter = Counter(words).most_common(vocab_size)

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


def process_str(s: str) -> str:
    s = s.lower()
    s = re.sub(r'([.,!?()"\'])', r' \1 ', s) # add whitespace around punctuation
    return s

def process_text(text: list[list[str,str]]) -> list[list[list[str],list[str]]]:
    for pair in text:
        pair[0] = f"{START_TOKEN} {process_str(pair[0])} {END_TOKEN}".split()
        pair[1] = f"{START_TOKEN} {process_str(pair[1])} {END_TOKEN}".split()
    return text

def load_data(datapath: str, shuffle=False) -> list[list[list[str],list[str]]]:
    with open(datapath) as f:
        data = f.readlines()
    data = [l.split('\t')[:2] for l in data]
    if shuffle:
        random.shuffle(data)
    return data

def generate_tokens(datapath: str, vocab_size: int, save=False):
    data = load_data(datapath, shuffle=True)
    data = process_text(data)

    vocab_en = Vocabulary()
    vocab_en.build([pair[0] for pair in data], vocab_size)
    vocab_de = Vocabulary()
    vocab_de.build([pair[1] for pair in data], vocab_size)

    data_tok = [(np.array(vocab_en.words_to_indices(pair[0])), 
                          np.array(vocab_de.words_to_indices(pair[1]))) for pair in data]
    if save:
        vocab_de.save("data/vocab_de.json")
        vocab_en.save("data/vocab_en.json")
        with h5py.File('data/data.h5', 'w') as f:
            dt = h5py.vlen_dtype(np.dtype('int32'))
            f.create_dataset('data_tok', data=data_tok, dtype=dt)
    return data_tok, vocab_en, vocab_de


def load_tokens(data_path: str, vocab_en_path: str, vocab_de_path: str):
    with h5py.File(data_path, 'r') as f:
        data_tok = f['data_tok'][:]
    vocab_en = Vocabulary.load(vocab_en_path)
    vocab_de = Vocabulary.load(vocab_de_path)
    return data_tok, vocab_en, vocab_de

# used as a collate function for the DataLoader
def prep_batch(batch):
    en_batch, ge_batch = zip(*batch)
    en_padded = pad_sequence(en_batch, batch_first=True, padding_value=0)  # corresponds to <PAD>
    en_mask = ~(en_padded == 0)
    ge_padded = pad_sequence(ge_batch, batch_first=True, padding_value=0)
    ge_mask = ~(ge_padded == 0)
    return ge_padded, ge_mask, en_padded, en_mask

def get_data(from_disk=True, batch_size=64, eval_frac=0.1, test_frac=0.05) -> (DataLoader, DataLoader, Vocabulary, Vocabulary):
    if from_disk:
        data_tok, vocab_en, vocab_de = load_tokens("data/data.h5", "data/vocab_en.json", "data/vocab_de.json")
    else:
        data_tok, vocab_en, vocab_de = generate_tokens("data/data.txt", 30000, save=True)
    data_tok = [(torch.from_numpy(pair[0]), torch.from_numpy(pair[1])) for pair in data_tok]

    train_high = int(len(data_tok)*(1-eval_frac-test_frac))
    eval_high = train_high + int(len(data_tok)*eval_frac)
    trainloader = DataLoader(data_tok[:train_high], batch_size=batch_size, collate_fn=prep_batch, shuffle=True)
    evalloader = DataLoader(data_tok[train_high:eval_high], batch_size=256, collate_fn=prep_batch, shuffle=True)
    testloader = DataLoader(data_tok[eval_high:], batch_size=10, collate_fn=prep_batch, shuffle=True)

    return trainloader, evalloader, testloader, vocab_en, vocab_de

