import pandas as pd

import feature_extraction
from feature_extraction import expand_rows

pd.set_option('display.max_columns', None)


lyrics_df = feature_extraction.load_and_process_lyrics('lyrics_train_set.csv')
expanded_lyrics = expand_rows(lyrics_df)
enriched_lyrics_df = feature_extraction.merge_lyrics_with_midi_features(expanded_lyrics, './midi_files')
print(enriched_lyrics_df.info())

print(enriched_lyrics_df.head())
