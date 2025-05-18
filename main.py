from gensim.models import KeyedVectors
from gensim.downloader import load
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from LyricsRNN import LyricsRNN

import feature_extraction
from LyricsDataset import LyricsDataset

from collections import Counter


def build_vocab(df, min_freq=1):
    """
    Build a vocabulary from the lyrics dataset.

    This function creates two mappings:
    1. word2idx: maps each word to a unique integer index
    2. idx2word: a list of words indexed by their corresponding ID

    These mappings are essential for next-word prediction tasks where:
    - Each target word (label) must be represented as an integer
    - Model predictions (indices) must be converted back to words

    The vocabulary includes:
    - All words that appear with frequency >= min_freq
    - Special tokens:
        <pad>: used for padding sequences
        <unk>: used to represent unknown or out-of-vocabulary words

    Args:
        df (pd.DataFrame): The dataframe containing the 'lyrics' column.
        min_freq (int): Minimum frequency a word must have to be included in the vocabulary.

    Returns:
        word2idx (dict): Dictionary mapping words to indices.
        idx2word (list): List mapping indices back to words.
    """

    counter = Counter()

    for line in df['lyrics']:
        tokens = line.strip().split()
        counter.update(tokens)

    # Filter rare words
    vocab_words = [word for word, freq in counter.items() if freq >= min_freq]

    # Special tokens
    special_tokens = ['<pad>', '<unk>']
    all_words = special_tokens + sorted(vocab_words)

    word2idx = {word: idx for idx, word in enumerate(all_words)}
    idx2word = all_words

    return word2idx, idx2word


word2vec = load("glove-wiki-gigaword-100")  # 100-dimensional GloVe

lyrics_df = feature_extraction.load_and_process_lyrics('lyrics_train_set.csv')
enriched_lyrics_df = feature_extraction.merge_lyrics_with_midi_features(lyrics_df, './midi_files')

word2idx, idx2word = build_vocab(enriched_lyrics_df, min_freq=2)

dataset = LyricsDataset(enriched_lyrics_df, word2vec, word2idx)

train_loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 100
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