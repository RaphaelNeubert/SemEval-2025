import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class TrainingConfig:
    device: str = 'cpu'
    learning_rate: float = 0.001
    num_epochs: int = 10
    log_interval: int = 100    # Steps between logging training loss
    eval_interval: int = 1000  # Steps between evaluations
    save_interval: int = 500   # Steps between saving model weights
    save_weights: bool = False
    save_weights_to: str = "./weights/weights.pth"
    disable_tqdm: bool = False

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

def training(config: TrainingConfig, model, trainloader, evalloader, log_writer=None):
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    steps = 0
    loss_accu = 0
    for epoch in range(config.num_epochs):
        for inputs, mask, targets in tqdm(trainloader, disable=config.disable_tqdm):
            inputs, mask, targets = inputs.to(device), mask.to(device), targets.to(device)
            opt.zero_grad()

            preds = model(inputs, mask)
            loss = torch.nn.BCEWithLogitsLoss()(preds, targets.to(torch.float))
            loss_accu += loss.item()
            loss.backward()
            opt.step()

            if (steps+1)%config.log_interval == 0:
                tqdm.write(f"train_loss: {loss_accu/100}")
                if log_writer is not None:
                    log_writer.add_scalar("training/loss", loss_accu/100, global_step=steps)
                loss_accu = 0 

            if steps%config.eval_interval == 0:
                eval_loss, eval_acc = evaluate(model, evalloader)
                tqdm.write(f"eval_loss: {eval_loss:.4f}, acc: {eval_acc:.4f}")
                if log_writer is not None:
                    log_writer.add_scalar("evaluation/loss", eval_loss, global_step=steps)
                    log_writer.add_scalar("evaluation/accuracy", eval_acc, global_step=steps)

            if steps%config.save_interval == 0:
                if config.save_weights:
                    torch.save(model.state_dict(), config.save_weights_to.replace("<training_step>", f"steps"))

            steps += 1
