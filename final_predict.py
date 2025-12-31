# final_gpt_predict.py
import os
import torch
import pandas as pd
from transformers import AutoTokenizer
from final_gpt_train import BertTextCNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained("chinese-roberta-wwm-ext")
    df = pd.read_csv("Atest.csv")
    texts = df["text"].astype(str).tolist()

    models = []
    for i in range(5):
        path = f"saved_models/final_gpt_fold{i}.pt"
        model = BertTextCNN("chinese-roberta-wwm-ext").to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)

    probs = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        with torch.no_grad():
            ps = []
            for m in models:
                p = torch.softmax(m(input_ids, attn), dim=1)[0, 1].item()
                ps.append(p)
        probs.append(sum(ps) / len(ps))

    df["label"] = [1 if p > 0.5 else 0 for p in probs]
    df["prob"] = probs
    df.to_csv("last_submit.csv", index=False, encoding="utf-8-sig")
    print("Saved last_submit.csv")

if __name__ == "__main__":
    main()
