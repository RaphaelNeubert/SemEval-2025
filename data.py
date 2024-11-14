from collections import Counter
from itertools import chain
import torch
#from model import PositionalEncoding
import re
import h5py
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

START_TOKEN = "<SOS>"
END_TOKEN = "<EOS>"
SPECIAL_TOKENS = ["<PAD>", "<UNK>"]     # special tokens get the first indices

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

def generate_vocabs(data: list[list[list[str],list[str]]], vocab_size: int):
    en_words = chain.from_iterable(pair[0] for pair in data)
    ge_words = chain.from_iterable(pair[1] for pair in data)

    en_counter = Counter(en_words).most_common(vocab_size)
    ge_counter = Counter(ge_words).most_common(vocab_size)

    en_word_to_index = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    ge_word_to_index = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

    en_word_to_index.update({word: idx + len(SPECIAL_TOKENS) for idx, (word, _) in enumerate(en_counter)})
    ge_word_to_index.update({word: idx + len(SPECIAL_TOKENS) for idx, (word, _) in enumerate(ge_counter)})

    en_index_to_word = list(en_word_to_index.keys())
    ge_index_to_word = list(ge_word_to_index.keys())

    return en_word_to_index, en_index_to_word, ge_word_to_index, ge_index_to_word

def words_to_indices(vocab: dict[str, int], words: list[str]) -> list[int]:
    return [vocab.get(word, vocab.get("<UNK>")) for word in words]

def indices_to_words(vocab: list[str], indices: list[int]) -> list[str]:
    return [vocab[idx] for idx in indices]

def generate_toks(datapath: str, vocab_size: int, save=False):
    data = load_data(datapath, shuffle=True)
    data = process_text(data)

    (en_word_to_index, en_index_to_word, 
     ge_word_to_index, ge_index_to_word) = generate_vocabs(data, vocab_size)

    data_tok = [(np.array(words_to_indices(en_word_to_index, pair[0])), 
                          np.array(words_to_indices(ge_word_to_index, pair[1]))) for pair in data]
    if save:
        with h5py.File('data/data.h5', 'w') as f:
            dt = h5py.vlen_dtype(np.dtype('int32'))
            f.create_dataset('data_tok', data=data_tok, dtype=dt)
            f.create_dataset('en_word_to_index', data=json.dumps(en_word_to_index), dtype=h5py.string_dtype())
            f.create_dataset('ge_word_to_index', data=json.dumps(ge_word_to_index), dtype=h5py.string_dtype())
            f.create_dataset('en_index_to_word', data=en_index_to_word, dtype=h5py.string_dtype())
            f.create_dataset('ge_index_to_word', data=ge_index_to_word, dtype=h5py.string_dtype())
    return data_tok, (en_word_to_index, en_index_to_word, ge_word_to_index, ge_index_to_word)

#data_tok, (en_word_to_index, en_index_to_word, ge_word_to_index, ge_index_to_word) = generate_toks("data/data.txt", 30000, save=True)

def load_toks(db_path):
    with h5py.File(db_path, 'r') as f:
        data_tok = f['data_tok'][:]
        en_word_to_index = json.loads(f['en_word_to_index'][()])
        ge_word_to_index = json.loads(f['ge_word_to_index'][()])
        en_index_to_word = list(f['en_index_to_word'][()])
        ge_index_to_word = list(f['ge_index_to_word'][()])
    return data_tok, (en_word_to_index, en_index_to_word, ge_word_to_index, ge_index_to_word)

data_tok, (en_word_to_index, en_index_to_word, ge_word_to_index, ge_index_to_word) = load_toks('data/data.h5')
data_tok = [(torch.from_numpy(pair[0]), torch.from_numpy(pair[1])) for pair in data_tok]

def prep_batch(batch):
    en_batch, ge_batch = zip(*batch)
    en_padded = pad_sequence(en_batch, batch_first=True, padding_value=0)  # corresponds to <PAD>
    en_mask = ~(en_padded == 0)
    ge_padded = pad_sequence(ge_batch, batch_first=True, padding_value=0)
    ge_mask = ~(ge_padded == 0)
    return ge_padded, ge_mask, en_padded, en_mask

dl = DataLoader(data_tok, batch_size=2, collate_fn=prep_batch, shuffle=True)

#print([indices_to_words(ge_index_to_word,sentence)for sentence in inputs])
inputs, input_mask, targets, target_mask = next(iter(dl))
print(inputs.shape)
print(targets.shape)
#print([indices_to_words(ge_index_to_word,sentence)for sentence in inputs])
#print(indices_to_words(ge_index_to_word,x[1][0]))
