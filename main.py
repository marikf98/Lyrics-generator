from gensim.models import KeyedVectors
from gensim.downloader import load
from torch.utils.data import DataLoader

import feature_extraction
from LyricsDataset import LyricsDataset

word2vec = load("glove-wiki-gigaword-100")  # 100-dimensional GloVe

lyrics_df = feature_extraction.load_and_process_lyrics('lyrics_train_set.csv')
enriched_lyrics_df = feature_extraction.merge_lyrics_with_midi_features(lyrics_df, './midi_files')

dataset = LyricsDataset(enriched_lyrics_df, word2vec)

train_loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

for batch in train_loader:
    print(batch['lyrics'].shape)         # torch.Size([32, 10, 100])
    print(batch['midi_features'].shape)  # torch.Size([32, 14])
    break

# # if __name__ == '__main__':
# #     print_hi('PyCharm')
#
