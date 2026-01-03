# final_train.py
import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel


# Utils 固定随机种子，保证实验的可重复性
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Dataset  数据预处理部分
class NewsDataset(Dataset):
    #初始化，准备原始数据，Dataset 接收原始文本、标签、分词器以及最大文本长度参数，为后续按需处理单条样本做准备。
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): #返回数据集样本数量，用于 DataLoader 控制迭代次数
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(  #使用 BERT 自带的 tokenizer 对原始文本进行分词和编码，而非人工分词。这样可以保证输入形式与预训练阶段保持一致，避免语义偏移。
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# BERT + TextCNN
class BertTextCNN(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 128, (k, hidden)) for k in [2, 3, 4]
        ])

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 3, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state.unsqueeze(1)  # [B,1,T,H]

        convs = []
        for conv in self.convs:
            c = torch.relu(conv(x)).squeeze(3)
            c = torch.max_pool1d(c, c.size(2)).squeeze(2)
            convs.append(c)

        x = torch.cat(convs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ======================
# Train
# ======================
def train(model, train_loader, val_loader, device, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_state = None

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attn)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        preds, probs, gts = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, attn)
                p = torch.softmax(logits, dim=1)[:, 1]

                preds.extend((p > 0.5).long().cpu().tolist())
                probs.extend(p.cpu().tolist())
                gts.extend(labels.cpu().tolist())

        acc = accuracy_score(gts, preds)
        auc = roc_auc_score(gts, probs)

        print(f"Epoch {ep}: ACC={acc:.4f} AUC={auc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

    return best_acc, best_state

# ======================
# Main
# ======================
def main(args):
    seed_everything() # 步骤1：固定随机种子（对应预处理的第三步）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(args.train_path)  # 步骤2：加载原始标注数据train.csv（对应预处理的第一步）

    texts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 单次划分（80 / 20）
    tr_idx, va_idx = train_test_split(
        range(len(texts)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_ds = NewsDataset(
        [texts[i] for i in tr_idx],
        [labels[i] for i in tr_idx],
        tokenizer,
        args.max_len
    )
    val_ds = NewsDataset(
        [texts[i] for i in va_idx],
        [labels[i] for i in va_idx],
        tokenizer,
        args.max_len
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = BertTextCNN(args.model_name).to(device)

    best_acc, best_state = train(
        model, train_loader, val_loader, device, args.epochs, args.lr
    )

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(best_state, f"{args.output_dir}/final_model.pt")

    print("\nFinal ACC:", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="train.csv")
    parser.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--output_dir", default="saved_models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)  # ✅ 256 → 128
    args = parser.parse_args()
    main(args)
