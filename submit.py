import pandas as pd
import re
import torch
import os
import zipfile

#pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 300)

def submit(model, tokenizer, label_set_thresholds):
    model.eval()
    device = next(model.parameters()).device
    emotions = ["anger", "fear", "joy", "sadness", "surprise"]
    df = pd.read_csv("data/public_data_dev/track_a/dev/eng.csv")
    for i, sentence in enumerate(df["text"]):
        #sentence_processed = re.sub(r'([.,!?()"\'])', r' \1 ', sentence.lower()).split()
        indices = torch.tensor(tokenizer.encode(sentence), device=device).unsqueeze(0)
        pred = torch.sigmoid(model(indices).squeeze())
        pred_classes = (pred > torch.tensor(label_set_thresholds, device=device)).to("cpu")
        for emo, label in zip(emotions, pred_classes):
            #print(label)
            df.loc[i, emo] = int(label.item())
    df[emotions] = df[emotions].astype(int)
    df = df.drop('text', axis=1)
    print(df)
    #print(df[["id", *emotions]])
    filename = "data/pred_eng.csv"
    df.to_csv(filename, index=False)

    with zipfile.ZipFile("data/submission.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(filename, arcname=("track_a/" + os.path.basename(filename)))
