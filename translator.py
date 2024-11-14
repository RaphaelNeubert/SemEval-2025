import torch
from data import get_data
#from torch.utils.data import TensorDataset, DataLoader
#from torch.nn.utils.rnn import pad_sequence


if __name__ == "__main__":
    trainloader, evalloader, vocab_en, vocab_de = get_data(from_disk=True, batch_size=5, eval_frac=0.2)

    inputs, input_mask, targets, target_mask = next(iter(trainloader))
    vocab_de.print_batch(inputs)
    print(input_mask)
