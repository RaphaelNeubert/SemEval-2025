import pandas as pd
import re
import torch
import os
import zipfile

#pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 300)

def submit(model, vocab):
    device = next(model.parameters()).device
    emotions = ["Anger", "Fear", "Joy", "Sadness", "Surprise"]
    df = pd.read_csv("data/eng_a.csv")
    for i, sentence in enumerate(df["text"]):
        sentence_processed = re.sub(r'([.,!?()"\'])', r' \1 ', sentence.lower()).split()
        indices = torch.tensor(vocab.words_to_indices(sentence_processed), device=device).unsqueeze(0)
        pred = torch.sigmoid(model(indices).squeeze())
        pred_classes = (pred > 0.5).to("cpu")
        for emo, label in zip(emotions, pred_classes):
            #print(label)
            df.loc[i, emo] = int(label.item())
    df[emotions] = df[emotions].astype(int)
    df = df.drop('text', axis=1)
    print(df)
    filename = "data/pred_eng_a.csv"
    df.to_csv(filename, index=False)

    with zipfile.ZipFile("data/submission.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(filename, arcname=os.path.basename(filename))
