import torch
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR


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
    unfreeze_count: int = 2    # pretraining only
    loss_label_weights: tuple[float] = (1,1,1,1,1)

def print_preds_batch(inputs, pred_classes, targets, tokenizer, writer=None, step=0):
    class_labels = ["anger", "fear", "joy", "sadness", "suprise"]
    text = ""
    for src, classes, target in zip(inputs, pred_classes, targets):
        input_indices = src[~(src == torch.tensor(tokenizer.pad_token_id, device=src.device))]
        input_sentence = " ".join(tokenizer.decode(input_indices))
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


def finetune_evaluate(model, dataloader, label_set_thresholds, print_test_evals=False, tokenizer=None, 
                      writer=None, step=0, disable_tqdm=False):
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_corr, total_samples = 0, 0, 0
    true_positives, false_positives, false_negatives = 0, 0, 0
    with torch.no_grad():
        for i, (inputs, mask, targets) in enumerate(tqdm(dataloader, disable=disable_tqdm)):
            inputs, mask, targets = inputs.to(device), mask.to(device), targets.to(device)
            preds = model(inputs, mask)
            loss = torch.nn.BCEWithLogitsLoss()(preds, targets.to(torch.float))
            total_loss += loss.item()

            probs = torch.sigmoid(preds)
            pred_classes = (probs > torch.tensor(label_set_thresholds, device=device)).to(torch.int)
            total_corr += (pred_classes == targets).all(dim=-1).sum().item()
            total_samples += targets.size(0)

            true_positives += ((pred_classes == 1) & (targets == 1)).sum().item()
            false_positives += ((pred_classes == 1) & (targets == 0)).sum().item()
            false_negatives += ((pred_classes == 0) & (targets == 1)).sum().item()

            if print_test_evals and i == 0:
                print_preds_batch(inputs, pred_classes, targets, tokenizer, writer, step)

    eval_loss = total_loss / len(dataloader)
    acc = total_corr / total_samples

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


    model.train()
    return eval_loss, acc, precision, recall, f1


def model_freeze(model, unfreeze_count: int):
    unfroozen_params = []
    for p in model.parameters():
        p.requires_grad = False
    for l in model.encoder.enc_layers[-unfreeze_count:]:
        for p in l.parameters():
            p.requires_grad = True
            unfroozen_params.append(p)
    for p in model.fc.parameters():
        p.requires_grad = True
        unfroozen_params.append(p)

    return unfroozen_params

def finetuning(config: TrainingConfig, model, trainloader, evalloader, label_set_thresholds,
               log_writer=None, print_test_evals=False, tokenizer=None, disable_tqdm=False):
    """
    if print_test_evals is set to True, tokenizer is expected to be not None
    """
    device = config.device
    unfroozen_params = model_freeze(model, config.unfreeze_count)
    opt = torch.optim.AdamW(unfroozen_params, lr=config.learning_rate)
    #opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.loss_label_weights, device=device))
    #lr_scheduler = LinearLR(opt, 1, 0.1, total_iters=20)
    steps = 0
    loss_accu = 0
    for epoch in range(config.num_epochs):
        for inputs, mask, targets in tqdm(trainloader, disable=disable_tqdm):
            inputs, mask, targets = inputs.to(device), mask.to(device), targets.to(device)
            opt.zero_grad()

            preds = model(inputs, mask)
            loss = loss_fn(preds, targets.to(torch.float))
            loss_accu += loss.item()
            loss.backward()
            opt.step()

            if steps%config.log_interval == 0 and steps > 0 :
                tqdm.write(f"train_loss: {loss_accu/config.log_interval}")
                if log_writer is not None:
                    log_writer.add_scalar("training/loss", loss_accu/config.log_interval, global_step=steps)
                loss_accu = 0 

            if steps%config.eval_interval == 0:
                eval_loss, eval_acc, precision, recall, f1 = finetune_evaluate(model, evalloader, label_set_thresholds, print_test_evals=print_test_evals,
                                                                               tokenizer=tokenizer, writer=log_writer, step=steps, disable_tqdm=disable_tqdm)
                tqdm.write(f"eval_loss: {eval_loss:.4f}, acc: {eval_acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1_score: {f1:.4f}")
                if log_writer is not None:
                    log_writer.add_scalar("evaluation/loss", eval_loss, global_step=steps)
                    log_writer.add_scalar("evaluation/accuracy", eval_acc, global_step=steps)
                    log_writer.add_scalar("evaluation/precision", precision, global_step=steps)
                    log_writer.add_scalar("evaluation/recall", recall, global_step=steps)
                    log_writer.add_scalar("evaluation/f1", f1, global_step=steps)

            if steps%config.save_interval == 0:
                if config.save_weights:
                    torch.save(model.state_dict(), config.save_weights_to.replace("<training_step>", f"{steps}"))

            steps += 1
        #lr_scheduler.step()
        #print("lr:", lr_scheduler.get_last_lr())

def pretrain_evaluate(model, evalloader, mask_token_id: int, label_mask_token_id: int, disable_tqdm=False):
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0
    total_corr = total_samples = 0 
    masked_correct = masked_total = 0 
    label_masked_correct = label_masked_total = 0
    for inputs, mask, targets in tqdm(evalloader, desc="evaluation", disable=disable_tqdm):
        inputs, mask, targets = inputs.to(device), mask.to(device), targets.to(device)
        with torch.no_grad():
            masked_tokens = (inputs ==  mask_token_id)
            label_masked_tokens = (inputs ==  label_mask_token_id)

            preds = model(inputs, mask)
            loss = torch.nn.CrossEntropyLoss()(preds.view(-1,preds.shape[-1]), targets.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(preds, dim=-1)
            corr = (preds == targets).all(dim=-1).sum()
            total_corr += corr
            total_samples += inputs.shape[0]

            masked_correct += (preds[masked_tokens] == targets[masked_tokens]).sum().item()
            masked_total += masked_tokens.sum()

            label_masked_correct += (preds[label_masked_tokens] == targets[label_masked_tokens]).sum().item()
            label_masked_total += label_masked_tokens.sum()

    eval_loss = total_loss / len(evalloader)
    acc_total = total_corr / total_samples
    acc_mask = masked_correct / masked_total if masked_total > 0 else 0
    acc_label_mask = label_masked_correct / label_masked_total if label_masked_total > 0 else 0
    model.train()
    return eval_loss, acc_total, acc_mask, acc_label_mask

def pretraining(config: TrainingConfig, model, trainloader, validloader, mask_token_id: int, label_mask_token_id: int, log_writer=None, disable_tqdm=False):
    device = config.device
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    steps = 0
    loss_accu = 0
    for epoch in range(config.num_epochs):
        for inputs, mask, targets in tqdm(trainloader, disable=disable_tqdm):
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
                eval_loss, eval_acc, eval_acc_masked, eval_acc_label_masked = pretrain_evaluate(model, validloader, mask_token_id, label_mask_token_id, disable_tqdm=disable_tqdm)
                tqdm.write(f"eval_loss: {eval_loss:.4f}, acc: {eval_acc:.4f}, acc_masked: {eval_acc_masked:.4f}, acc_label_masked: {eval_acc_label_masked:.4f}")
                if log_writer is not None:
                    log_writer.add_scalar("preevaluation/loss", eval_loss, global_step=steps)
                    log_writer.add_scalar("preevaluation/accuracy", eval_acc, global_step=steps)
                    log_writer.add_scalar("preevaluation/accuracy_masked", eval_acc_masked, global_step=steps)
                    log_writer.add_scalar("preevaluation/accuracy_label_mask", eval_acc_label_masked, global_step=steps)

            if steps%config.save_interval == 0:
                if config.save_weights:
                    torch.save(model.state_dict(), config.save_weights_to.replace("<training_step>", f"{steps}"))

            steps += 1 
