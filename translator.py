import torch
from data import get_data
from model import Transformer
#from torch.utils.data import TensorDataset, DataLoader
#from torch.nn.utils.rnn import pad_sequence


if __name__ == "__main__":
    trainloader, evalloader, vocab_en, vocab_de = get_data(from_disk=True, batch_size=5, eval_frac=0.2)

    inputs, input_mask, targets, target_mask = next(iter(trainloader))
    #vocab_de.print_batch(inputs)

    model = Transformer(src_vocab_size=vocab_de.size(), tgt_vocab_size=vocab_en.size(), dim_embeddings=100, num_heads=5,
                        ffn_hidden_dims=100*4, num_encoder_layers=3, num_decoder_layers=3)
    model(inputs, targets, src_mask=input_mask, tgt_mask=target_mask)
