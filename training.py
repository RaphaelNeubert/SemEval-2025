import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass


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

def print_preds_batch(inputs, pred_classes, targets, vocab, writer=None, step=0):
    class_labels = ["anger", "fear", "joy", "sadness", "suprise"]
    text = ""
    for src, classes, target in zip(inputs, pred_classes, targets):
        input_indices = src[~(src == torch.tensor(vocab.words_to_indices([vocab.pad_token]), device=src.device))]
        input_sentence = " ".join(vocab.index_to_words(input_indices))
        prediction_words = " ".join([class_labels[i] for i in range(len(classes)) if classes[i]==1])
        target_words = " ".join([class_labels[i] for i in range(len(classes)) if target[i]==1])
        example = (f"**Input:** {input_sentence}\n"
                   f"**Prediction:** {prediction_words}\n"
                   f"**Target:** {target_words}\n"
                    "-----------------------------\n")
        text += example
    #print(text)
    if writer is not None:
        writer.add_text(f"Example emotion detections", text, global_step=step)


def evaluate(model, dataloader, print_test_evals=False, vocab=None, writer=None, step=0):
    model.eval()
    total_loss = 0
    total_corr = 0
    total_samples = 0
    with torch.no_grad():
        for i, (inputs, mask, targets) in enumerate(dataloader):
            inputs, mask, targets = inputs.to(device), mask.to(device), targets.to(device)
            preds = model(inputs, mask)
            loss = torch.nn.BCEWithLogitsLoss()(preds, targets.to(torch.float))
            total_loss += loss.item()

            probs = torch.sigmoid(preds)
            pred_classes = (probs > 0.5).to(torch.int)
            total_corr += (pred_classes == targets).all(dim=-1).sum().item()
            total_samples += targets.shape[0]

            if i == 0:
                print_preds_batch(inputs, pred_classes, targets, vocab, writer, step)

    eval_loss = total_loss / len(dataloader)
    acc = total_corr / total_samples
    model.train()
    return eval_loss, acc

def training(config: TrainingConfig, model, trainloader, evalloader, log_writer=None, print_test_evals=False, vocab=None):
    """
    if print_test_evals is set to True, vocab is expected to be not None
    """
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

            if steps%config.log_interval == 0 and steps > 0 :
                tqdm.write(f"train_loss: {loss_accu/config.log_interval}")
                if log_writer is not None:
                    log_writer.add_scalar("training/loss", loss_accu/config.log_interval, global_step=steps)
                loss_accu = 0 

            if steps%config.eval_interval == 0:
                eval_loss, eval_acc = evaluate(model, evalloader, print_test_evals=print_test_evals, 
                                               vocab=vocab, writer=log_writer, step=steps)
                tqdm.write(f"eval_loss: {eval_loss:.4f}, acc: {eval_acc:.4f}")
                if log_writer is not None:
                    log_writer.add_scalar("evaluation/loss", eval_loss, global_step=steps)
                    log_writer.add_scalar("evaluation/accuracy", eval_acc, global_step=steps)

            if steps%config.save_interval == 0:
                if config.save_weights:
                    torch.save(model.state_dict(), config.save_weights_to.replace("<training_step>", f"{steps}"))

            steps += 1

def pretrain_evaluate(model, evalloader, disable_tqdm=False):
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0
    total_corr = 0
    total_samples = 0
    for inputs, mask, targets in tqdm(evalloader, desc="evaluation", disable=disable_tqdm):
        inputs, mask, targets = inputs.to(device), mask.to(device), targets.to(device)
        with torch.no_grad():
            preds = model(inputs, mask)
            loss = torch.nn.CrossEntropyLoss()(preds.view(-1,preds.shape[-1]), targets.view(-1))
            total_loss += loss.item()
            preds = torch.argmax(preds, dim=-1)
            corr = (preds == targets).all(dim=-1).sum()
            total_corr += corr
            total_samples += inputs.shape[0]

    eval_loss = total_loss / len(evalloader)
    acc = total_corr / total_samples
    model.train()
    return eval_loss, acc

def pretraining(config: TrainingConfig, model, trainloader, validloader, log_writer=None):
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    steps = 0
    loss_accu = 0
    for epoch in range(config.num_epochs):
        for inputs, mask, targets in tqdm(trainloader, disable=config.disable_tqdm):
            inputs, mask, targets = inputs.to(device), mask.to(device), targets.to(device)
            opt.zero_grad()

            preds = model(inputs, mask)
            loss = torch.nn.CrossEntropyLoss()(preds.view(-1,preds.shape[-1]), targets.view(-1))
            loss_accu += loss.item()
            loss.backward()
            opt.step()

            if steps%config.log_interval == 0 and steps > 0 :
                tqdm.write(f"train_loss: {loss_accu/config.log_interval}")
                if log_writer is not None:
                    log_writer.add_scalar("pretraining/loss", loss_accu/config.log_interval, global_step=steps)
                loss_accu = 0 

            if steps%config.eval_interval == 0:
                eval_loss, eval_acc = pretrain_evaluate(model, validloader, disable_tqdm=config.disable_tqdm)
                tqdm.write(f"eval_loss: {eval_loss:.4f}, acc: {eval_acc:.4f}")
                if log_writer is not None:
                    log_writer.add_scalar("preevaluation/loss", eval_loss, global_step=steps)
                    log_writer.add_scalar("preevaluation/accuracy", eval_acc, global_step=steps)

            if steps%config.save_interval == 0:
                if config.save_weights:
                    torch.save(model.state_dict(), config.save_weights_to.replace("<training_step>", f"{steps}"))

            steps += 1 
