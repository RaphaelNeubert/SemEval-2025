import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_corr = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, mask, targets in dataloader:
            inputs, mask, targets = inputs.to(device), mask.to(device), targets.to(device)
            preds = model(inputs, mask)
            loss = torch.nn.BCEWithLogitsLoss()(preds, targets.to(torch.float))
            total_loss += loss.item()

            probs = torch.sigmoid(preds)
            pred_classes = (probs > 0.5).to(torch.int)
            total_corr += (pred_classes == targets).all(dim=-1).sum().item()
            total_samples += targets.shape[0]

    eval_loss = total_loss / len(dataloader)
    acc = total_corr / total_samples
    model.train()
    return eval_loss, acc

def training(model):
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

            if steps%1000 == 0:
                eval_loss, eval_acc = evaluate(model, evalloader)
                tqdm.write(f"eval_loss: {eval_loss:.4f}, acc: {eval_acc:.4f}")
                writer.add_scalar("evaluation/loss", eval_loss, global_step=steps)
                writer.add_scalar("evaluation/accuracy", eval_acc, global_step=steps)

            if steps%500 == 0:
                if SAVE_WEIGHTS:
                    torch.save(model.state_dict(), SAVE_WEIGHTS_TO)

            steps += 1
