import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SAINT(nn.Module):
    def __init__(self, num_concepts, num_interactions, d_model=128, nhead=4, num_layers=2):
        super(SAINT, self).__init__()
        self.d_model = d_model
        
        # Encoder: Context (Content IDs, Difficulty, etc.)
        self.concept_embedding = nn.Embedding(num_concepts, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder: Behavior (Response types, Latency, Hover counts)
        self.behavior_embedding = nn.Embedding(num_interactions, d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final Latent Vector (The Personality Vector)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, context_seq, behavior_seq):
        # 1. Process Context through Encoder
        enc_emb = self.pos_encoder(self.concept_embedding(context_seq))
        memory = self.transformer_encoder(enc_emb)

        # 2. Process Behavior through Decoder (attending to Context memory)
        dec_emb = self.pos_decoder(self.behavior_embedding(behavior_seq))
        out = self.transformer_decoder(dec_emb, memory)

        # 3. Extract the last token's representation as the Personality Vector
        personality_vector = self.fc(out[:, -1, :])
        return personality_vector