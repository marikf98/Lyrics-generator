from torch.utils.data import Dataset
import torch
import numpy as np

class LyricsDataset(Dataset):
    def __init__(self, lyrics_df, word2vec_model, word2idx, max_words=10):
        self.lyrics_df = lyrics_df.reset_index(drop=True)
        self.word2vec = word2vec_model
        self.word2idx = word2idx
        self.max_words = max_words
        self.embedding_dim = word2vec_model.vector_size
        self.samples = []

        for _, row in self.lyrics_df.iterrows():
            tokens = row['lyrics'].strip().split()

            if len(tokens) < 2:
                continue  # not enough to create (input, target)

            for i in range(1, len(tokens)):
                input_words = tokens[:i][-self.max_words:]  # max N previous words
                target_word = tokens[i]

                input_vec = self._vectorize_words(input_words)
                midi_vec = self._extract_midi_features(row)
                target_idx = self.word2idx.get(target_word, self.word2idx['<unk>'])

                if not isinstance(input_vec, torch.Tensor) or not isinstance(midi_vec, torch.Tensor):
                    if not isinstance(input_vec, torch.Tensor):
                        print(f"Invalid input_vec for lyrics: {row['lyrics']}")
                    if not isinstance(midi_vec, torch.Tensor):
                        print(f"Invalid midi_vec for band={row['band']} song={row['song']}")
                    continue

                # if input_vec is None or midi_vec is None or target_idx is None:
                #     print("Skipping a sample with missing data.")
                #     continue

                self.samples.append({
                    'input_seq': input_vec,
                    'midi_features': midi_vec,
                    'target_idx': target_idx
                })

    def _vectorize_words(self, words):
        vectors = []

        for word in words:
            try:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    if vec is not None and isinstance(vec, (list, np.ndarray)):
                        vectors.append(torch.tensor(vec, dtype=torch.float32))
                    else:
                        raise ValueError(f"Vector for word '{word}' is malformed.")
                else:
                    vectors.append(torch.zeros(self.embedding_dim))
            except Exception as e:
                print(f"Error vectorizing word '{word}': {e}")
                vectors.append(torch.zeros(self.embedding_dim))

        # Pad with zeros at the beginning
        while len(vectors) < self.max_words:
            vectors.insert(0, torch.zeros(self.embedding_dim))

        # Truncate if too long
        if len(vectors) > self.max_words:
            vectors = vectors[-self.max_words:]

        try:
            return torch.stack(vectors)
        except Exception as e:
            print(f"Failed to stack vectors. Returning None. Error: {e}")
            return None

    def _extract_midi_features(self, row):
        try:
            midi_feats = []
            for key in [
                'num_of_instruments', 'num_of_drums', 'pitch_range', 'most_frequent_pitch',
                'melodic_interval_mean', 'melodic_interval_std', 'tempo',
                'avg_note_duration', 'onset_density', 'avg_velocity',
                'key_signature', 'time_sig_numerator', 'time_sig_denominator'
            ]:
                val = row.get(key, None)
                if val is None:
                    return None  # Invalidate this row
                midi_feats.append(float(val))
            return torch.tensor(midi_feats, dtype=torch.float32)
        except Exception as e:
            print(f"Error extracting MIDI vector: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]