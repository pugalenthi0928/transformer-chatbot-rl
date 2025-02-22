import torch
import torch.nn as nn
import torch.optim as optim

# Sample conversational pairs (input -> output)
pairs = [
    ("hello", "hi"),
    ("how are you", "i am fine"),
    ("bye", "goodbye"),
    ("thanks", "welcome")
]

# Build vocabulary from the pairs (including special tokens)
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
vocab = {SOS_TOKEN: 0, EOS_TOKEN: 1}
for inp, out in pairs:
    for word in (inp + " " + out).split():
        if word not in vocab:
            vocab[word] = len(vocab)
vocab_size = len(vocab)

# Helper to encode a sentence into token indices with sos/eos
def encode_sentence(sentence):
    tokens = [SOS_TOKEN] + sentence.split() + [EOS_TOKEN]
    return [vocab[t] for t in tokens]

# Prepare training data as lists of token indices
train_data = []
for inp, out in pairs:
    src_indices = encode_sentence(inp)
    tgt_indices = encode_sentence(out)
    train_data.append((src_indices, tgt_indices))

# Define positional encoding module for the transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                              (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Define the Transformer-based Seq2Seq model
class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                         num_encoder_layers=num_layers,
                                         num_decoder_layers=num_layers,
                                         dim_feedforward=64, 
                                         batch_first=True)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt):
        src_emb = self.pos_enc(self.embedding(src))
        tgt_emb = self.pos_enc(self.embedding(tgt))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(src.device)
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output

# Initialize model, loss, and optimizer
model = TransformerChatbot(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
model.train()
for epoch in range(100):
    for src_indices, tgt_indices in train_data:
        src = torch.tensor(src_indices).unsqueeze(0)
        tgt = torch.tensor(tgt_indices).unsqueeze(0)
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]
        output = model(src, tgt_input)
        output = output.permute(0, 2, 1)
        loss = criterion(output, tgt_expected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Inference: test the chatbot
model.eval()
def generate_reply(sentence):
    src = torch.tensor(encode_sentence(sentence)).unsqueeze(0)
    tgt_input = torch.tensor([vocab[SOS_TOKEN]]).unsqueeze(0)
    result = []
    for _ in range(10):
        output = model(src, tgt_input)
        next_token = output[0, -1].argmax().item()
        if next_token == vocab[EOS_TOKEN]:
            break
        result.append(next_token)
        tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]])], dim=1)
    return " ".join([list(vocab.keys())[list(vocab.values()).index(idx)] for idx in result])

# Test chatbot
print("User: hello")
print("Bot:", generate_reply("hello"))
print("User: how are you")
print("Bot:", generate_reply("how are you"))
print("User: thanks")
print("Bot:", generate_reply("thanks"))
