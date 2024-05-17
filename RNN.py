import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# 数据预处理和分割
df = pd.read_csv('Modified_SQL_Dataset.csv')
df['Label'] = df['Label'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(df['Query'], df['Label'], test_size=0.2, random_state=42)


# 分词器
def tokenizer(text):
    return text.split()


# 构建词汇表
def build_vocab(texts):
    token_counts = Counter()
    for text in texts:
        token_counts.update(tokenizer(text))
    vocab = {token: i + 1 for i, token in enumerate(token_counts)}
    vocab['<unk>'] = 0
    return vocab


vocab = build_vocab(X_train)


# 文本到序列的转换函数
def text_to_sequence(text, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer(text)]


# 自定义 Dataset 类
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [torch.tensor(text_to_sequence(text, vocab), dtype=torch.long) for text in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# 为 DataLoader 定义 collate_fn
def collate_batch(batch):
    label_list, text_list = [], []
    for _text, _label in batch:
        label_list.append(_label)
        text_list.append(_text)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    return text_list, torch.tensor(label_list)


train_dataset = TextDataset(X_train.to_list(), y_train.to_list(), vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
test_dataset = TextDataset(X_test.to_list(), y_test.to_list(), vocab)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)


# 定义 RNN 模型
class RNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(RNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.gru(embedded)
        return torch.sigmoid(self.fc(hidden.squeeze(0)))


model = RNNTextClassifier(len(vocab) + 1, 50, 128)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 训练和评估模型
def train_model(model, train_loader, test_loader, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss, total_acc = 0, 0
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = loss_fn(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += ((outputs.squeeze() > 0.5) == labels).float().mean().item()
        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_acc / len(train_loader)

        model.eval()
        total_test_acc = 0
        with torch.no_grad():
            for texts, labels in test_loader:
                outputs = model(texts)
                total_test_acc += ((outputs.squeeze() > 0.5) == labels).float().mean().item()
        avg_test_acc = total_test_acc / len(test_loader)

        print(
            f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Test Acc: {avg_test_acc:.4f}')


# 执行训练
train_model(model, train_loader, test_loader, 10)
