import os

import pretty_midi
import pandas as pd
import numpy as np
from collections import Counter


def load_and_process_lyrics(lyrics_path):
    """
    Load and process the lyrics from a CSV file with columns: band, song, lyrics.
    Spaces are preserved; only lowercasing and stripping is applied for matching.
    """
    df = pd.read_csv(lyrics_path, sep=',', header=None, usecols=[0, 1, 2])
    df.columns = ['band', 'song', 'lyrics']
    # Standardize for matching
    df['band'] = df['band'].str.strip().str.lower()
    df['song'] = df['song'].str.strip().str.lower()
    return df

def midi_filename_to_band_song(midi_filename):
    base = midi_filename[:-4]
    if "-_" not in base:
        return None, None
    band_part, song_part = base.split("-_", 1)
    band = band_part.replace('_', ' ').lower().strip()
    song = song_part.replace('_', ' ').lower().strip()
    return band, song
def extract_midi_features(midi_path):
    features = {}

    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error reading MIDI file {midi_path}: {e}")
        return None

    # Instruments
    try:
        features['num_of_instruments'] = len(midi_data.instruments)
    except Exception as e:
        print(f"Could not extract num_of_instruments from {midi_path}: {e}")
        features['num_of_instruments'] = 0 #Fallback - consider other values

    try:
        features['num_of_drums'] = sum(inst.is_drum for inst in midi_data.instruments)
    except Exception as e:
        print(f"Could not extract num_of_drums from {midi_path}: {e}")
        features['num_of_drums'] = 0 #Fallback - consider other values

    # Notes and derived features
    try:
        notes = [note for inst in midi_data.instruments for note in inst.notes]
        pitches = [note.pitch for note in notes]
        velocities = [note.velocity for note in notes]
        note_durations = [note.end - note.start for note in notes]
    except Exception as e:
        print(f"Could not extract notes/pitches/velocities from {midi_path}: {e}")
        notes, pitches, velocities, note_durations = [], [], [], []

    try:
        features['pitch_histogram'] = list(np.histogram(pitches, bins=range(128), density=True)[0]) if pitches else [0.0]*128
    except Exception as e:
        print(f"Could not extract pitch_histogram from {midi_path}: {e}")
        features['pitch_histogram'] = [0.0]*128

    try:
        features['most_frequent_pitch'] = int(np.argmax(features['pitch_histogram'])) if features['pitch_histogram'] is not None else 0
    except Exception as e:
        print(f"Could not extract most_frequent_pitch from {midi_path}: {e}")
        features['most_frequent_pitch'] = 0 #Fallback - consider other values

    try:
        features['pitch_range'] = int(max(pitches) - min(pitches)) if pitches else 0
    except Exception as e:
        print(f"Could not extract pitch_range from {midi_path}: {e}")
        features['pitch_range'] = 0 #Fallback - consider other values

    try:
        melodic_intervals = np.diff(pitches) if len(pitches) > 1 else [0.0]
        features['melodic_interval_mean'] = float(np.mean(melodic_intervals)) if melodic_intervals is not None else  0.0
        features['melodic_interval_std'] = float(np.std(melodic_intervals)) if melodic_intervals is not None else  0.0
    except Exception as e:
        print(f"Could not extract melodic intervals from {midi_path}: {e}")
        features['melodic_interval_mean'] = 0.0 #Fallback - consider other values
        features['melodic_interval_std'] = 0.0 #Fallback - consider other values

    try:
        features['tempo'] = midi_data.estimate_tempo() if midi_data.get_end_time() > 0 else  0.0
    except Exception as e:
        print(f"Could not extract tempo from {midi_path}: {e}")
        features['tempo'] = 0.0 #Fallback - consider other values

    try:
        features['avg_note_duration'] = float(np.mean(note_durations)) if note_durations else  0.0
    except Exception as e:
        print(f"Could not extract avg_note_duration from {midi_path}: {e}")
        features['avg_note_duration'] = 0.0 #Fallback - consider other values

    try:
        features['onset_density'] = len(pitches) / midi_data.get_end_time() if midi_data.get_end_time() > 0 else  0.0
    except Exception as e:
        print(f"Could not extract onset_density from {midi_path}: {e}")
        features['onset_density'] = 0.0 #Fallback - consider other values

    try:
        features['velocity_histogram'] = list(np.histogram(velocities, bins=range(128), density=True)[0]) if velocities else [0.0]*128
    except Exception as e:
        print(f"Could not extract velocity_histogram from {midi_path}: {e}")
        features['velocity_histogram'] = [0.0]*128 #Fallback - consider other values

    try:
        features['avg_velocity'] = float(np.mean(velocities)) if velocities else  0.0
    except Exception as e:
        print(f"Could not extract avg_velocity from {midi_path}: {e}")
        features['avg_velocity'] = 0.0 #Fallback - consider other values

    return features
def merge_lyrics_with_midi_features(lyrics_df, midi_dir):
    lyrics_pairs = set(zip(lyrics_df['band'].str.lower().str.strip(), lyrics_df['song'].str.lower().str.strip()))
    midi_fail_counter=0
    error_counter=0

    # Build a lookup for fast assignment: {(band, song): features}
    midi_features = {}
    for filename in os.listdir(midi_dir):
        if not filename.lower().endswith('.mid'):
            continue
        band, song = midi_filename_to_band_song(filename)
        if band is None or song is None:
            print(f"Skipped malformed MIDI filename: {filename}")
            midi_fail_counter+=1
            continue
        if (band, song) in lyrics_pairs:
            features = extract_midi_features(os.path.join(midi_dir, filename))
            if features is not None:
                midi_features[(band, song)] = features
            else:
                error_counter+=1
        else:
            print(f"MIDI files not found in lyrics CSV: {filename}")
            midi_fail_counter += 1
    print(f"Midi not found in csv counter: {midi_fail_counter}")
    print(f"Error in Midi file: {error_counter}")

    # Initialize feature columns
    feature_keys = next(iter(midi_features.values())).keys() if midi_features else []
    for key in feature_keys:
        lyrics_df[key] = None

    # Fill features in the DataFrame
    for (band, song), features in midi_features.items():
        mask = (lyrics_df['band'] == band) & (lyrics_df['song'] == song)
        for key, value in features.items():
            if isinstance(value, list):
                for idx in lyrics_df[mask].index:
                    lyrics_df.at[idx, key] = value
            else:
                lyrics_df.loc[mask, key] = value

    return lyrics_df

def expand_rows (lyrics_df):
    expanded_rows = []
    for idx, row in lyrics_df.iterrows():
        band = row['band']
        song = row['song']
        lyrics_lines = [line.strip() for line in str(row['lyrics']).split('&') if line.strip()]
        for line in lyrics_lines:
            expanded_rows.append({'band': band, 'song': song, 'lyrics': line})

    lyrics_lines_df = pd.DataFrame(expanded_rows)
    return lyrics_lines_df



def build_vocab(df, min_freq=1):
    counter = Counter()

    for line in df['lyrics']:
        tokens = line.strip().lower().split()
        counter.update(tokens)

    # Filter rare words
    vocab_words = [word for word, freq in counter.items() if freq >= min_freq]

    # Special tokens
    special_tokens = ['<pad>', '<unk>']
    all_words = special_tokens + sorted(vocab_words)

    word2idx = {word: idx for idx, word in enumerate(all_words)}
    idx2word = all_words

    return word2idx, idx2word


def drop_rows_missing_midi_features(df, midi_cols=None):

    before = len(df)
    if midi_cols is None:
        midi_cols = [c for c in df.columns if c not in ['band', 'song', 'lyrics']]

    # Mask for missing MIDI features
    missing_mask = df[midi_cols].isnull().any(axis=1)
    dropped_rows = df[missing_mask]
    dropped_songs = dropped_rows[['band', 'song']].drop_duplicates()

    print(f"Dropped {len(dropped_rows)} lyric lines with missing MIDI features.")
    print(f"Dropped {len(dropped_songs)} unique songs with missing MIDI features.")
    if not dropped_songs.empty:
        print("Dropped songs:")
        print(dropped_songs.to_string(index=False))

    # Keep only rows WITHOUT missing MIDI
    cleaned_df = df[~missing_mask].reset_index(drop=True)
    after = len(cleaned_df)
    print(f"{before - after} lyric lines dropped. {after} lyric lines remaining.")
    return cleaned_df