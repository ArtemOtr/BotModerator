import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from data_preproc import df_train, df_test, X2, y2
from collections import Counter
import re
from tqdm import trange, tqdm

def custom_tokenizer(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens




learning_rate = 1e-4  

nepochs = 1
def text_to_indices(text, vocab, max_len):
    default_index = vocab['<unk>']
    tokens = custom_tokenizer(text)
    indices = [vocab.get(token, default_index) for token in tokens]
    
    indices = [vocab["<sos>"]] + indices[:max_len - 2] + [vocab["<eos>"]]
    indices += [vocab["<pad>"]] * (max_len - len(indices))
    return indices

class CustomDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        indices = text_to_indices(text, self.vocab, self.max_len)
        return torch.tensor(indices), torch.tensor(label)


class LSTM(nn.Module):
    def __init__(self, num_emb, output_size, num_layers=1, hidden_size=128):
        super(LSTM, self).__init__()

        
        self.embedding = nn.Embedding(num_emb, hidden_size)

        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.5)

        
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_in, mem_in):
        
        input_embs = self.embedding(input_seq)

        
        output, (hidden_out, mem_out) = self.lstm(input_embs, (hidden_in, mem_in))

        
        return self.fc_out(output), hidden_out, mem_out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


hidden_size = 64
num_layers = 3





df_train_1 = df_train.sample(frac = 1).reset_index(drop=True)
df_test_1 = df_test.sample(frac = 1).reset_index(drop=True)



texts_train_1 = df_train_1['text'].tolist()
label_train_1 = df_train_1['label'].tolist()
texts_test_1 = df_test_1['text'].tolist()
label_test_1 = df_test_1['label'].tolist()


texts_combined_1 = texts_train_1 + texts_test_1
labels_combined_1 = label_train_1 + label_test_1

texts_combined_2 = X2
labels_combined_2 = y2
texts_combined_2 = texts_combined_2.reset_index(drop=True)
labels_combined_2 = labels_combined_2.reset_index(drop=True)
texts_combined_2 = list(texts_combined_2)
labels_combined_2 = list(labels_combined_2)

counter_1 = Counter()
for text in texts_combined_1:
    counter_1.update(custom_tokenizer(text))

vocab_1 = {word: idx for idx, (word, _) in enumerate(counter_1.most_common(), start=4)}
vocab_1["<pad>"] = 0
vocab_1["<sos>"] = 1
vocab_1["<eos>"] = 2
vocab_1["<unk>"] = 3

counter_2 = Counter()
for text in texts_combined_2:
    counter_2.update(custom_tokenizer(text))

vocab_2 = {word: idx for idx, (word, _) in enumerate(counter_2.most_common(), start=4)}
vocab_2["<pad>"] = 0
vocab_2["<sos>"] = 1
vocab_2["<eos>"] = 2
vocab_2["<unk>"] = 3

max_len_1 = max(len(text.split()) for text in texts_combined_1)
max_len_2 = max(len(text.split()) for text in texts_combined_2)

train_dataset_1 = CustomDataset(texts_combined_1, labels_combined_1, vocab_1, max_len_1)
train_dataset_2 = CustomDataset(texts_combined_2, labels_combined_2, vocab_2, max_len_2)

train_dataloader_1 = DataLoader(train_dataset_1, batch_size=16, shuffle=True)
train_dataloader_2 = DataLoader(train_dataset_2, batch_size=16, shuffle=True)

lstm_classifier_1 = LSTM(num_emb=len(vocab_1), output_size=2,
                         num_layers=num_layers, hidden_size=hidden_size).to(device)
lstm_classifier_2 = LSTM(num_emb=len(vocab_2), output_size=2,
                         num_layers=num_layers, hidden_size=hidden_size).to(device)

# Optimizers for both models
optimizer_1 = optim.Adam(lstm_classifier_1.parameters(), lr=learning_rate)
optimizer_2 = optim.Adam(lstm_classifier_2.parameters(), lr=learning_rate)

# Loss function
loss_fn = nn.CrossEntropyLoss()

training_loss_logger_1 = []
training_loss_logger_2 = []
training_acc_logger_1 = []
training_acc_logger_2 = []
if __name__ == '__main__':
    pbar_1 = trange(0, nepochs, leave=False, desc="Epoch 1")
    for epoch in pbar_1:
        pbar_1.set_postfix_str('Training Epoch 1')

        lstm_classifier_1.train()
        train_loss_1, train_correct_1, train_total_1 = 0, 0, 0

        for text, label in tqdm(train_dataloader_1, desc="Training 1", leave=False):
            text = text.to(device)
            label = label.to(device)

            bs = text.size(0)
            hidden = torch.zeros(num_layers, bs, hidden_size, device=device)
            memory = torch.zeros(num_layers, bs, hidden_size, device=device)

            pred, hidden, memory = lstm_classifier_1(text, hidden, memory)
            pred = pred[:, -1, :]

            loss = loss_fn(pred, label)

            optimizer_1.zero_grad()
            loss.backward()
            optimizer_1.step()

            train_loss_1 += loss.item()
            train_correct_1 += (pred.argmax(dim=1) == label).sum().item()
            train_total_1 += bs

        train_acc_1 = train_correct_1 / train_total_1
        training_loss_logger_1.append(train_loss_1 / len(train_dataloader_1))
        training_acc_logger_1.append(train_acc_1)

    # Training loop for the second model
    pbar_2 = trange(0, nepochs, leave=False, desc="Epoch 2")
    for epoch in pbar_2:
        pbar_2.set_postfix_str('Training Epoch 2')

        lstm_classifier_2.train()
        train_loss_2, train_correct_2, train_total_2 = 0, 0, 0

        for text, label in tqdm(train_dataloader_2, desc="Training 2", leave=False):
            text = text.to(device)
            label = label.to(device)

            bs = text.size(0)
            hidden = torch.zeros(num_layers, bs, hidden_size, device=device)
            memory = torch.zeros(num_layers, bs, hidden_size, device=device)

            pred, hidden, memory = lstm_classifier_2(text, hidden, memory)
            pred = pred[:, -1, :]

            loss = loss_fn(pred, label)

            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()

            train_loss_2 += loss.item()
            train_correct_2 += (pred.argmax(dim=1) == label).sum().item()
            train_total_2 += bs

        train_acc_2 = train_correct_2 / train_total_2
        training_loss_logger_2.append(train_loss_2 / len(train_dataloader_2))
    training_acc_logger_2.append(train_acc_2)

    torch.save(lstm_classifier_1.state_dict(), 'C:/Users/nenad/TgBot/lstm_model_toxic.pth')
    torch.save(lstm_classifier_2.state_dict(), 'C:/Users/nenad/TgBot/lstm_model_spam.pth')

    print("Both models have been trained and saved!")
