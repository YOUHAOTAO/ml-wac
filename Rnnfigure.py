import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Data preprocessing and splitting
df = pd.read_csv('Modified_SQL_Dataset.csv')
df['Label'] = df['Label'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(df['Query'], df['Label'], test_size=0.2, random_state=42)

# Tokenizer
def tokenizer(text):
    return text.split()

# Build vocabulary
def build_vocab(texts):
    token_counts = Counter()
    for text in texts:
        token_counts.update(tokenizer(text))
    vocab = {token: i + 1 for i, token in enumerate(token_counts)}
    vocab['<unk>'] = 0
    return vocab

vocab = build_vocab(X_train)

# Text to sequence conversion function
def text_to_sequence(text, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer(text)]

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [torch.tensor(text_to_sequence(text, vocab), dtype=torch.long) for text in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Collate function for DataLoader
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

# Define the RNN model
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

# Train and evaluate the model
def train_model(model, train_loader, test_loader, epochs):
    train_losses, train_accs, test_accs = [], [], []
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
        train_losses.append(total_loss / len(train_loader))
        train_accs.append(total_acc / len(train_loader))

        model.eval()
        total_test_acc = 0
        with torch.no_grad():
            for texts, labels in test_loader:
                outputs = model(texts)
                total_test_acc += ((outputs.squeeze() > 0.5) == labels).float().mean().item()
        test_accs.append(total_test_acc / len(test_loader))

        print(f'Epoch {epoch + 1}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Test Acc: {test_accs[-1]:.4f}')

    # Plot training and testing accuracy and loss curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot([1 - acc for acc in train_accs], label='Train Error')
    plt.title('Training Loss and Error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Error')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Execute training
train_model(model, train_loader, test_loader, 10)
