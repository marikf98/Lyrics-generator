from torch.utils.data import Dataset
import torch


class LyricsDataset(Dataset):
    def __init__(self, lyrics_df, word2vec_model, max_words=10):
        self.lyrics_df = lyrics_df.reset_index(drop=True)
        self.word2vec = word2vec_model
        self.embedding_dim = word2vec_model.vector_size
        self.max_words = max_words
        self.samples = []

        for _, row in self.lyrics_df.iterrows():
            lyric = row['lyrics']
            lyric_embedding = self._vectorize_lyric(lyric)
            midi_features = self._extract_midi_features(row)

            if lyric_embedding is not None and midi_features is not None:
                self.samples.append({
                    'lyrics': lyric_embedding,
                    'midi_features': midi_features
                })

    def _vectorize_lyric(self, text):
        tokens = text.strip().split()
        vectors = []

        for word in tokens[:self.max_words]:
            if word in self.word2vec:
                vectors.append(torch.tensor(self.word2vec[word], dtype=torch.float32))
            else:
                vectors.append(torch.zeros(self.embedding_dim))

        # Pad if needed
        while len(vectors) < self.max_words:
            vectors.append(torch.zeros(self.embedding_dim))

        return torch.stack(vectors)

    def _extract_midi_features(self, row):
        try:
            features = [
                row['num_of_instruments'],
                row['num_of_drums'],
                row['most_frequent_pitch'],
                row['pitch_range'],
                row['melodic_interval_mean'],
                row['melodic_interval_std'],
                row['tempo'],
                row['avg_note_duration'],
                row['onset_density'],
                row['avg_velocity'],
                row['key_signature'],
                1 if row['mode'] == 'major' else 0,
                row['time_sig_numerator'],
                row['time_sig_denominator']
            ]
            return torch.tensor(features, dtype=torch.float32)
        except:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]