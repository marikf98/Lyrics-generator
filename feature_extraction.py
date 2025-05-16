import os
import re

import pretty_midi
import pandas as pd
import numpy as np

def load_and_process_lyrics(lyrics_path):
    """
    Load and process the lyrics from a CSV file using comma as the delimiter.

    Args:
        lyrics_path (str): Path to the CSV file containing lyrics.

    Returns:
        pd.DataFrame: A DataFrame with columns ['band', 'song', 'lyrics'], one row per lyric line.
    """
    # Comma-delimited file with no header
    df = pd.read_csv(lyrics_path, sep=',', header=None, usecols=[0, 1, 2])
    df.columns = ['band', 'song', 'lyrics']

    records = []
    for _, row in df.iterrows():
        band = row['band'].strip().lower()
        song = row['song'].strip().lower()
        full_lyrics = str(row['lyrics']).strip().lower()

        lines = full_lyrics.strip(' &').split(' & ')
        for line in lines:
            if line.strip():
                records.append({'band': band, 'song': song, 'lyrics': line.strip()})

    return pd.DataFrame(records)
def extract_midi_features(midi_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        # Number of instruments
        num_of_instruments = len(midi_data.instruments)
        num_of_drums = sum(inst.is_drum for inst in midi_data.instruments)

        # Number of drum tracks
        num_of_drums = sum(1 for instrument in midi_data.instruments if instrument.is_drum)

        # Pitch histogram
        pitches = [note.pitch for instrument in midi_data.instruments for note in instrument.notes]
        pitch_histogram, _ = np.histogram(pitches, bins=range(128), density=True)
        pitch_histogram = pitch_histogram.tolist()

        # Pitch range
        pitch_range = max(pitches) - min(pitches) if pitches else 0

        # Most frequent pitch
        most_frequent_pitch = np.argmax(pitch_histogram) if pitches else 0

        #pitch range
        pitch_range = max(pitches) - min(pitches) if pitches else 0

        # Melodic intervals
        melodic_intervals = np.diff(pitches) if len(pitches) > 1 else np.array([0])
        melodic_interval_mean = float(np.mean(melodic_intervals))
        melodic_interval_std = float(np.std(melodic_intervals))

        # Tempo
        tempo = midi_data.estimate_tempo()
        note_durations = [note.end - note.start for inst in midi_data.instruments for note in inst.notes]
        avg_note_duration = float(np.mean(note_durations)) if note_durations else 0
        onset_density = len(pitches) / midi_data.get_end_time() if midi_data.get_end_time() > 0 else 0

        # Note durations
        note_durations = [note.end - note.start for instrument in midi_data.instruments for note in instrument.notes]
        avg_note_duration = np.mean(note_durations) if note_durations else 0

        # Onset density
        total_time = midi_data.get_end_time()
        onset_density = len(pitches) / total_time if total_time > 0 else 0

        # Velocity histogram
        velocities = [note.velocity for inst in midi_data.instruments for note in inst.notes]
        velocity_histogram, _ = np.histogram(velocities, bins=range(128), density=True)
        avg_velocity = float(np.mean(velocities)) if velocities else 0

        # Average velocity
        avg_velocity = np.mean(velocities) if velocities else 0

        # Key signature
        key_signature = midi_data.key_signature_changes[0].key_number if midi_data.key_signature_changes else 0
        mode = 1 if key_signature >= 0 else 0  # 1=major, 0=minor

        # Mode (major or minor)
        mode = 'major' if key_signature is not None and key_signature >= 0 else 'minor'

        # Time signature
        time_signature = midi_data.time_signature_changes[0] if midi_data.time_signature_changes else None

        numerator = time_signature.numerator if time_signature else 4
        denominator = time_signature.denominator if time_signature else 4

        # Return extracted features
        return {
            'num_of_instruments': num_of_instruments,
            'num_of_drums': num_of_drums,
            'pitch_histogram': pitch_histogram,
            'most_frequent_pitch': most_frequent_pitch,
            'pitch_range': pitch_range,
            'melodic_interval_mean': melodic_interval_mean,
            'melodic_interval_std': melodic_interval_std,
            'tempo': tempo,
            'avg_note_duration': avg_note_duration,
            'onset_density': onset_density,
            'velocity_histogram': velocity_histogram,
            'avg_velocity': avg_velocity,
            'key_signature': key_signature,
            'mode': mode,
            'time_sig_numerator': numerator,
            'time_sig_denominator': denominator
        }

    except Exception as e:
        print(f"Error processing MIDI file {midi_path}: {e}")
        return None


def merge_lyrics_with_midi_features(lyrics_df, midi_dir):

    # Initialize columns
    for col in [
        'num_of_instruments', 'num_of_drums', 'pitch_histogram', 'most_frequent_pitch', 'pitch_range',
        'melodic_interval_mean', 'melodic_interval_std', 'tempo', 'avg_note_duration', 'onset_density',
        'velocity_histogram', 'avg_velocity', 'key_signature', 'mode', 'time_sig_numerator', 'time_sig_denominator'
    ]:
        lyrics_df[col] = None

    for filename in os.listdir(midi_dir):

        if not filename.endswith('.mid'):
            continue

        # file_path = os.path.join(midi_dir, filename)
        # base_name = os.path.splitext(filename)[0] # Remove the file extension
        # if '-' not in base_name:
        #     continue
        # band_part, song_part = base_name.split('-', 1)
        # band_name = band_part.replace('_', ' ').strip().lower()
        # song_name = song_part.replace('_', ' ').strip().lower()

        file_path = os.path.join(midi_dir, filename)
        base_name = os.path.splitext(filename)[0]  # Remove .mid extension

        # Use regex to split on patterns like _-_, -_, _-, - with optional underscores/spaces
        match = re.split(r'[_ ]*-[ _]*', base_name, maxsplit=1)

        if len(match) != 2:
            print(f"Skipping due to unmatched pattern: {base_name}")
            continue

        band_name = match[0].replace('_', ' ').strip().lower()
        song_name = match[1].replace('_', ' ').strip().lower()

        mask = (lyrics_df['band'] == band_name) & (lyrics_df['song'] == song_name)
        if not mask.any():
            if not mask.any():
                print(f"No match found for: {band_name} - {song_name}")
                continue

        midi_features = extract_midi_features(file_path)

        if midi_features is None:
            continue

        for key, value in midi_features.items():
            if isinstance(value, (list, np.ndarray)):
                for idx in lyrics_df[mask].index:
                    lyrics_df.at[idx, key] = value
            else:
                lyrics_df.loc[mask, key] = value

    return lyrics_df