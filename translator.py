import torch
import torch.nn.functional as F
from data import get_data
from model import Transformer
from tqdm import tqdm
#from torch.utils.data import TensorDataset, DataLoader
#from torch.nn.utils.rnn import pad_sequence

torch.set_printoptions(profile="full")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.tensorboard import SummaryWriter

SAVE_WEIGHTS = False
SAVE_WEIGHTS_TO = "data/weights1.pth"
LOAD_WEIGHTS = True
LOAD_WEIGHTS_FROM = "data/weights1.pth"

def evaluate(model, evalloader):
    model.eval()
    words_corr = 0
    num_words = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, input_mask, targets, target_mask in evalloader:
            inputs, input_mask  = inputs.to(device), input_mask.to(device)
            targets, target_mask = targets.to(device), target_mask.to(device)
            pred = model(inputs, targets[:,:-1], src_mask=input_mask, tgt_mask=target_mask[:,:-1])
            targets = targets[:,1:]
            eval_loss = torch.nn.CrossEntropyLoss(ignore_index=0)(pred.view(-1,pred.shape[-1]), targets.reshape(-1).to(torch.long))
            total_loss += eval_loss.item()
            pred_indices = torch.argmax(pred, dim=-1)
            mask = (targets != 0) # padding tokens
            words_corr += torch.sum((pred_indices == targets) & mask)
            num_words += torch.sum(mask)

    acc = words_corr/num_words
    model.train()
    return total_loss/len(evalloader), acc

def test_translations(step, model, dataloader, src_vocab, tgt_vocab, writer=None):
    model.eval()  
    dont_print = ["<PAD>","<SOS>", "<EOS>"]
    with torch.no_grad():
        for inputs, input_mask, targets, target_mask in dataloader:
            inputs, input_mask  = inputs.to(device), input_mask.to(device)
            targets, target_mask = targets.to(device), target_mask.to(device)
            prediction = model(inputs, targets[:,:-1], src_mask=input_mask, tgt_mask=target_mask[:,:-1])
            prediction = torch.argmax(prediction, dim=-1)
            print(targets.shape)
            print(prediction.shape)

            text_log = ""
            for src, pred, target in zip(inputs, prediction, targets):
                src_indices = src[~torch.isin(src, torch.tensor(src_vocab.words_to_indices(dont_print), device=device))]
                pred_indices = pred[~torch.isin(pred, torch.tensor(tgt_vocab.words_to_indices(dont_print), device=device))]
                target_indices = target[~torch.isin(target, torch.tensor(tgt_vocab.words_to_indices(dont_print), device=device))]
                input_sentence = " ".join(src_vocab.index_to_words(src_indices))
                pred_sentence = " ".join(tgt_vocab.index_to_words(pred_indices))
                target_sentence = " ".join(tgt_vocab.index_to_words(target_indices))
                example = (f"**Input:** {input_sentence}\n"
                           f"**Prediction:** {pred_sentence}\n"
                           f"**Target:** {target_sentence}\n"
                            "-----------------------------\n")
                text_log += example
            print(text_log)
            if writer is not None:
                writer.add_text(f"Translation Example", text_log, global_step=step)


            break   # one random batch is enough for now
    model.train()  


if __name__ == "__main__":
    writer = SummaryWriter()
    trainloader, evalloader, testloader, vocab_en, vocab_de = get_data(from_disk=True, batch_size=64, eval_frac=0.05, test_frac=0.01)

    model = Transformer(src_vocab_size=vocab_de.size(), tgt_vocab_size=vocab_en.size(), dim_embeddings=100, num_heads=5,
                        ffn_hidden_dims=100*4, num_encoder_layers=3, num_decoder_layers=3, dropout=0.1).to(device)

    if LOAD_WEIGHTS:
        model.load_state_dict(torch.load(LOAD_WEIGHTS_FROM, weights_only=True))

    opt = torch.optim.Adam(model.parameters())

    #print("test")
    #src = "Ich habe gewonnen ! .".split(" ")
    #start_idx = vocab_en.words_to_indices(["<SOS>"])[0]
    #end_idx = vocab_en.words_to_indices(["<EOS>"])[0]
    #src_indices = torch.tensor(vocab_de.words_to_indices(src), dtype=torch.long, device=device).unsqueeze(0)
    #tgt_indices = model.inference(src_indices, start_idx, end_idx, max_len=200)
    #print(tgt_indices.shape)
    #print(" ".join(vocab_en.index_to_words(tgt_indices.squeeze())))

    steps = 0
    loss_accu = 0
    for epoch in range(1000):
        for inputs, input_mask, targets, target_mask in tqdm(trainloader):
            inputs, input_mask  = inputs.to(device), input_mask.to(device)
            targets, target_mask = targets.to(device), target_mask.to(device)
            opt.zero_grad()

            pred = model(inputs, targets[:,:-1], src_mask=input_mask, tgt_mask=target_mask[:,:-1])
            loss = torch.nn.CrossEntropyLoss()(pred.view(-1,pred.shape[-1]), targets[:,1:].reshape(-1).to(torch.long))
            #loss = F.cross_entropy(pred, targets[:,1:].to(torch.long))
            loss_accu += loss.item()
            loss.backward()
            opt.step()

            if (steps+1)%100 == 0:
                tqdm.write(f"train_loss: {loss_accu/100}")
                writer.add_scalar("training/loss", loss_accu/100, global_step=steps)
                loss_accu = 0 

            if steps%200 == 0:
                eval_loss, eval_acc = evaluate(model, evalloader)
                tqdm.write(f"eval_loss: {eval_loss}, acc: {eval_acc}")
                writer.add_scalar("evaluation/loss", eval_loss, global_step=steps)
                writer.add_scalar("evaluation/accuracy", eval_acc, global_step=steps)

            if steps%500 == 0:
                test_translations(steps, model, testloader, vocab_de, vocab_en, writer=writer)
                if SAVE_WEIGHTS:
                    torch.save(model.state_dict(), SAVE_WEIGHTS_TO)

            steps += 1

