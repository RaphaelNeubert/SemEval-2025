import torch
import torch.nn.functional as F
from data import get_data
from model import Transformer, SemModel
from tqdm import tqdm

torch.set_printoptions(profile="full")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.tensorboard import SummaryWriter

SAVE_WEIGHTS = False
SAVE_WEIGHTS_TO = "data/weights1.pth"
LOAD_WEIGHTS = False
LOAD_WEIGHTS_FROM = "data/weights1.pth"


if __name__ == "__main__":
    writer = SummaryWriter()
    trainloader, evalloader, testloader, vocab = get_data(batch_size=64)

    model = SemModel(vocab_size=vocab.size(), num_classes=5, dim_embeddings=512, 
                     num_heads=8, ffn_hidden_dims=512*4, num_encoder_layers=6, dropout=0.1).to(device)


    if LOAD_WEIGHTS:
        model.load_state_dict(torch.load(LOAD_WEIGHTS_FROM, weights_only=True))

    opt = torch.optim.Adam(model.parameters())

    steps = 0
    loss_accu = 0
    for epoch in range(1):
        for inputs, mask, targets in tqdm(trainloader):
            inputs, mask, targets = inputs.to(device), mask.to(device), targets.to(device)
            opt.zero_grad()

            preds = model(inputs, mask)
            loss = torch.nn.BCEWithLogitsLoss()(preds, targets.to(torch.float))
            loss_accu += loss.item()
            loss.backward()
            opt.step()

            if (steps+1)%100 == 0:
                tqdm.write(f"train_loss: {loss_accu/100}")
                writer.add_scalar("training/loss", loss_accu/100, global_step=steps)
                loss_accu = 0 

            #if steps%200 == 0:
            #    eval_loss, eval_acc = evaluate(model, evalloader)
            #    tqdm.write(f"eval_loss: {eval_loss}, acc: {eval_acc}")
            #    writer.add_scalar("evaluation/loss", eval_loss, global_step=steps)
            #    writer.add_scalar("evaluation/accuracy", eval_acc, global_step=steps)

            if steps%500 == 0:
                if SAVE_WEIGHTS:
                    torch.save(model.state_dict(), SAVE_WEIGHTS_TO)

            steps += 1

