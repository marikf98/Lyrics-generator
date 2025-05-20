from gensim.models import KeyedVectors
from gensim.downloader import load
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from LyricsRNN import LyricsRNN

import feature_extraction
from LyricsDataset import LyricsDataset
import gensim.downloader as api




word2vec = api.load("word2vec-google-news-300")
VECTOR_SIZE = 300

lyrics_df = feature_extraction.load_and_process_lyrics('lyrics_train_set.csv')
expanded_lyrics = feature_extraction.expand_rows(lyrics_df)
enriched_lyrics_df = feature_extraction.merge_lyrics_with_midi_features(expanded_lyrics, './midi_files')
enriched_lyrics_df = feature_extraction.drop_rows_missing_midi_features(enriched_lyrics_df)


word2idx, idx2word = feature_extraction.build_vocab(enriched_lyrics_df, min_freq=2)

dataset = LyricsDataset(enriched_lyrics_df, word2vec, word2idx)

train_loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

#=================================== up to here

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 300
hidden_dim = 128
midi_dim = 14
vocab_size = len(word2idx)

model = LyricsRNN(embedding_dim, hidden_dim, midi_dim, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        input_seq = batch['input_seq'].to(device)         # [batch_size, 10, 100]
        midi_feats = batch['midi_features'].to(device)    # [batch_size, 14]
        target_idx = batch['target_idx'].to(device)       # [batch_size]

        optimizer.zero_grad()
        logits = model(input_seq, midi_feats)             # [batch_size, vocab_size]
        loss = criterion(logits, target_idx)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}")

model.eval()
with torch.no_grad():
    batch = next(iter(train_loader))
    input_seq = batch['input_seq'].to(device)
    midi_feats = batch['midi_features'].to(device)

    logits = model(input_seq, midi_feats)
    preds = torch.argmax(logits, dim=1)

    print("Predicted indices:", preds[:5].tolist())
    print("Corresponding words:", [idx2word[i] for i in preds[:5].tolist()])