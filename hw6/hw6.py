import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split

from datetime import datetime as dt
import time, os

# root = './' <- why use this?
root = os.path.dirname(os.path.abspath(__file__))

# set device
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""

You can implement any necessary methods.

"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_Q, d_K, d_V, numhead, dropout):    
      super().__init__()
      
      #Q1. Implement
      self.numhead = numhead
      self.d_Q = d_Q
      self.d_K = d_K
      self.d_V = d_V

      self.W_Q = nn.Linear(d_model, d_Q * numhead)
      self.W_K = nn.Linear(d_model, d_K * numhead)
      self.W_V = nn.Linear(d_model, d_V * numhead)

      self.out_proj = nn.Linear(d_V * numhead, d_model)
      self.dropout = nn.Dropout(dropout)
      
    
    def forward(self, x_Q, x_K, x_V, src_batch_lens=None):
      
      # Q2. Implement
      bsz, q_len, _ = x_Q.shape
      k_len = x_K.shape[1]

      Q = self.W_Q(x_Q).view(bsz, q_len, self.numhead, self.d_Q).transpose(1, 2)
      K = self.W_K(x_K).view(bsz, k_len, self.numhead, self.d_K).transpose(1, 2)
      V = self.W_V(x_V).view(bsz, k_len, self.numhead, self.d_V).transpose(1, 2)

      scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_K)

      if src_batch_lens is not None:
        mask = torch.arange(k_len, device=scores.device).expand(bsz, k_len)
        mask = mask >= src_batch_lens.unsqueeze(1)
        mask = mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask, float('-inf'))

      attn = torch.softmax(scores, dim=-1)
      attn = self.dropout(attn)

      ctx = torch.matmul(attn, V)
      ctx = ctx.transpose(1, 2).contiguous().view(bsz, q_len, self.numhead * self.d_V)
      out = self.out_proj(ctx)
      out = self.dropout(out)

      return out

class TF_Encoder_Block(nn.Module):
    def __init__(self, d_model, d_ff, numhead, dropout):    
      super().__init__()
    
      # Q3. Implment constructor for transformer encoder block
      self.self_attn = MultiHeadAttention(d_model, d_model//numhead, d_model//numhead, d_model//numhead, numhead, dropout)
      self.dropout = nn.Dropout(dropout)
      self.norm1 = nn.LayerNorm(d_model)
      self.feed_forward = nn.Sequential(
          nn.Linear(d_model, d_ff),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(d_ff, d_model)
      )
      self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_batch_lens):
      
        # Q4. Implment forward function for transformer encoder block
        attn_out = self.self_attn(x, x, x, src_batch_lens)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        out = self.norm2(x + self.dropout(ff_out))
        return out

"""
Positional encoding
PE(pos,2i) = sin(pos/10000**(2i/dmodel))
PE(pos,2i+1) = cos(pos/10000**(2i/dmodel))
"""

def PosEncoding(t_len, d_model, *, device=None, dtype=None):
    """Standard sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    if device is None:
        device = 'cpu'
    if dtype is None:
        dtype = torch.float32

    position = torch.arange(t_len, device=device, dtype=dtype).unsqueeze(1)  # (t_len, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_model)
    )  # (d_model/2,)

    pe = torch.zeros(t_len, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class TF_Encoder(nn.Module):
    def __init__(self, vocab_size, d_model,
                 d_ff, numlayer, numhead, dropout, pad_idx=None):    
        super().__init__()
        
        self.numlayer = numlayer
        self.src_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )
        self.dropout=nn.Dropout(dropout)

        # Q5. Implement a sequence of numlayer encoder blocks
        self.enc_blocks = nn.ModuleList([TF_Encoder_Block(d_model, d_ff, numhead, dropout) for _ in range(numlayer)])
        
    def forward(self, x, src_batch_lens):

                x_embed = self.src_embed(x)
                x = self.dropout(x_embed)
                p_enc = PosEncoding(x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)
                x = x + p_enc

                # Q6. Implement: forward over numlayer encoder blocks
                for block in self.enc_blocks:
                        x = block(x, src_batch_lens)

                return x



"""

main model

"""

class sentiment_classifier(nn.Module):
    
    def __init__(self, enc_input_size, 
                 enc_d_model,
                 enc_d_ff,
                 enc_num_layer,
                 enc_num_head,
                 dropout,
                 pad_idx,
                ):    
        super().__init__()

        self.pad_idx = pad_idx
        
        self.encoder = TF_Encoder(
            vocab_size=enc_input_size,
            d_model=enc_d_model,
            d_ff=enc_d_ff,
            numlayer=enc_num_layer,
            numhead=enc_num_head,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=enc_d_model, out_features=enc_d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=enc_d_model, out_features=1),
        )
          
   
    def forward(self, x, x_lens):
        # encoder output: (B, T, D)
        src_ctx = self.encoder(x, src_batch_lens=x_lens)

        # masked mean pooling over non-pad tokens
        mask = (x != self.pad_idx).float().unsqueeze(-1)  # (B, T, 1)
        summed = (src_ctx * mask).sum(dim=1)              # (B, D)
        denom = mask.sum(dim=1).clamp(min=1.0)            # (B, 1)
        pooled = summed / denom                           # (B, D)

        out_logits = self.classifier(pooled).squeeze(-1)  # (B,)
        return out_logits

"""

datasets

"""

# Load IMDB dataset
# you should have the file 'imdb_dataset_dict.pt' in your local directory

# imdb_dataset_dict_path = './imdb_dataset_dict.pt'
imdb_dataset_dict_path = os.path.join(root, 'imdb_dataset_dict.pt')

# load dataset
imdb_dataset_dict =torch.load(imdb_dataset_dict_path, weights_only=False)

# get train/test datasets
train_dataset = imdb_dataset_dict[0] 
test_dataset = imdb_dataset_dict[1]
# get character dictionary
src_word_dict = imdb_dataset_dict[2]
src_idx_dict = imdb_dataset_dict[3]

split_ratio = 0.85
num_train = int(len(train_dataset) * split_ratio)
split_train, split_valid = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# Set hyperparam (batch size)
batch_size_trn = 128
batch_size_val = 256
batch_size_tst = 256

train_dataloader = DataLoader(split_train, batch_size=batch_size_trn, shuffle=True)
val_dataloader = DataLoader(split_valid, batch_size=batch_size_val, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size_tst, shuffle=False)

SRC_PAD_IDX = src_word_dict['<PAD>']

# show sample reviews with pos/neg sentiments

show_sample_reviews = True

if show_sample_reviews:
    
    sample_text, sample_lab = next(iter(train_dataloader))
    slist=[]

    for stxt in sample_text[:4]: 
        slist.append([src_idx_dict[j] for j in stxt])

    for j, s in enumerate(slist):
        print('positive' if sample_lab[j]==1 else 'negative')
        print(' '.join([i for i in s if i != '<PAD>'])+'\n')


"""

model

"""

enc_vocab_size = len(src_word_dict) # counting eof, one-hot vector goes in

# Set hyperparam (model size)
# Examples: 
# model & ff dim - 4, 8, 16, 32, 64, 128
# numhead & numlayer 1~4

enc_d_model = 64

enc_d_ff = 128

enc_num_head = 4

enc_num_layer= 2

DROPOUT=0.1

model = sentiment_classifier(enc_input_size=enc_vocab_size,
                         enc_d_model = enc_d_model,     
                         enc_d_ff = enc_d_ff, 
                         enc_num_head = enc_num_head, 
                         enc_num_layer = enc_num_layer,
                         dropout=DROPOUT,
                         pad_idx=SRC_PAD_IDX) 

model = model.to(dev)

"""

optimizer

"""

# Set hyperparam (learning rate)
# examples: 1e-3 ~ 1e-5

lr = 1e-3

weight_decay = 1e-2
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Reduce LR when validation loss plateaus
# (PyTorch version compatibility: some versions don't support `verbose`.)
try:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        threshold=1e-4,
        verbose=True,
    )
except TypeError:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        threshold=1e-4,
    )

criterion = nn.BCEWithLogitsLoss()

"""

auxiliary functions

"""


# get length of reviews in batch
def get_lens_from_tensor(x):
    # lens (batch, t)
    lens = torch.ones_like(x).long()
    lens[x==SRC_PAD_IDX]=0
    return torch.sum(lens, dim=-1)

def get_binary_metrics(y_pred, y):
    # find number of TP, TN, FP, FN
    TP=sum(((y_pred == 1)&(y==1)).type(torch.int32))
    FP=sum(((y_pred == 1)&(y==0)).type(torch.int32))
    TN=sum(((y_pred == 0)&(y==0)).type(torch.int32))
    FN=sum(((y_pred == 0)&(y==1)).type(torch.int32))
    accy = (TP+TN)/(TP+FP+TN+FN)
            
    recall = TP/(TP+FN) if TP+FN!=0 else 0
    prec = TP/(TP+FP) if TP+FP!=0 else 0
    f1 = 2*recall*prec/(recall+prec) if recall+prec !=0 else 0
    
    return accy, recall, prec, f1

"""

train/validation

""" 


def train(model, dataloader, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(dataloader):

        src = batch[0].to(dev)
        trg = batch[1].float().to(dev)

        # print('batch trg.shape', trg.shape)
        # print('batch src.shape', src.shape)

        optimizer.zero_grad()

        x_lens = get_lens_from_tensor(src).to(dev)

        output = model(x=src, x_lens=x_lens) 


        output = output.contiguous().view(-1)
        trg = trg.contiguous().view(-1)
        
        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):

    model.eval()
    
    epoch_loss = 0
    
    epoch_accy =0
    epoch_recall =0
    epoch_prec =0
    epoch_f1 =0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            src = batch[0].to(dev)
            trg = batch[1].float().to(dev)

            x_lens = get_lens_from_tensor(src).to(dev)

            output = model(x=src, x_lens=x_lens) 

            output = output.contiguous().view(-1)
            trg = trg.contiguous().view(-1)

            loss = criterion(output, trg)
            
            accy, recall, prec, f1 = get_binary_metrics((output>=0).long(), trg.long())
            epoch_accy += accy
            epoch_recall += recall
            epoch_prec += prec
            epoch_f1 += f1

            epoch_loss += loss.item()

    # show accuracy
    print(f'\tAccuracy: {epoch_accy/(len(dataloader)):.3f}')
    
    return epoch_loss / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

"""

Training loop

"""

N_EPOCHS = 100
CLIP = 1

best_valid_loss = float('inf')
ckpt_path = os.path.join(root, 'model.pt')

# Early stopping (validation loss)
EARLY_STOP_PATIENCE = 8
early_stop_counter = 0

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_dataloader, criterion)

    # step scheduler on validation loss
    scheduler.step(valid_loss)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), ckpt_path)
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    if early_stop_counter >= EARLY_STOP_PATIENCE:
        print(f'Early stopping triggered (patience={EARLY_STOP_PATIENCE}).')
        break
        
"""

Test loop

"""
print('*** Now test phase begins! ***')
model.load_state_dict(torch.load(ckpt_path))

test_loss = evaluate(model, test_dataloader, criterion)

print(f'| Test Loss: {test_loss:.3f}')
