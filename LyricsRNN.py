import torch
import torch.nn as nn

class LyricsRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, midi_dim, vocab_size, num_layers=1):
        """
        Initializes the RNN-based model for next-word prediction conditioned on MIDI features.

        Args:
            embedding_dim (int): Dimensionality of the word embeddings (e.g., 100 for GloVe).
            hidden_dim (int): Number of hidden units in the GRU.
            midi_dim (int): Number of MIDI features used for conditioning (e.g., 14).
            vocab_size (int): Size of the output vocabulary; used to define the output layer.
            num_layers (int): Number of stacked GRU layers (default = 1).

        Model Flow:
            - input_seq (lyrics): [batch_size, seq_len, embedding_dim]
            - midi_features:      [batch_size, midi_dim]
            - GRU encodes the lyrics → [batch_size, hidden_dim]
            - Concatenate GRU output with MIDI features → [batch_size, hidden_dim + midi_dim]
            - Final fully connected layers project to vocab size
              → [batch_size, vocab_size] for next-word prediction.
        """
        super().__init__()

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + midi_dim, 256),  # Input: combined GRU + MIDI features
            nn.ReLU(),  # Non-linear activation
            nn.Linear(256, vocab_size)  # Output: logits for each word in the vocabulary
        )

    def forward(self, input_seq, midi_features):

        # input_seq: [batch_size, seq_len, embedding_dim]
        # midi_features: [batch_size, midi_dim]

        """
        Forward pass of the model.

        Args:
            input_seq (Tensor): Embedded lyrics of shape [batch_size, seq_len, embedding_dim].
            midi_features (Tensor): MIDI conditioning features of shape [batch_size, midi_dim].

        Returns:
            logits (Tensor): Output scores for each word in the vocabulary,
                             shape [batch_size, vocab_size].

        Process:
            - Pass input_seq through GRU → extract final hidden state [batch_size, hidden_dim]
            - Concatenate with MIDI features → [batch_size, hidden_dim + midi_dim]
            - Pass through fully connected layers → [batch_size, vocab_size]
        """

        _, h_n = self.rnn(input_seq)  # h_n: [num_layers, batch, hidden_dim]
        h_final = h_n[-1]             # [batch, hidden_dim]

        combined = torch.cat([h_final, midi_features], dim=1)  # [batch, hidden + midi]
        logits = self.fc(combined)     # [batch, vocab_size]
        return logits