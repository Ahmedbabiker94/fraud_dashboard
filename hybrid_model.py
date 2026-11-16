"""Original Hybrid GNN-VAE-GRU architecture and scoring from the user's notebook.

Requires torch-geometric. Scores depend on batch composition, and are NOT calibrated probabilities.
No labels are passed to the model; placeholder labels are used only to construct Data.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GraphConv


class FeatureAttention(nn.Module):
    def __init__(self, in_features):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        att_weights = self.attention(x)
        return x * att_weights

class EnhancedHybridGNNVAEGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, latent_dim, heads=4, dropout=0.3):
        super(EnhancedHybridGNNVAEGRU, self).__init__()

        # Feature attention
        self.feature_attention = FeatureAttention(in_channels)

        # Multi-layer GNN with residual connections
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=1)
        self.conv2 = GCNConv(hidden_channels * heads, hidden_channels * heads)
        self.conv3 = GraphConv(hidden_channels * heads, out_channels)

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)
        self.norm3 = nn.LayerNorm(out_channels)

        # VAE components
        self.fc_mu = nn.Sequential(
            nn.Linear(out_channels, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(out_channels, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # Bidirectional GRU for temporal modeling
        self.gru = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim // 2, 1)
        )

        # Auxiliary classifier
        self.aux_classifier = nn.Sequential(
            nn.Linear(out_channels, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1)
        )

        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, in_channels)
        )

    def encode(self, x, edge_index, edge_attr):
        # Apply feature attention
        x = self.feature_attention(x)

        # First graph convolution
        x1 = self.norm1(F.leaky_relu(self.conv1(x, edge_index, edge_attr), 0.2))

        # Second graph convolution with residual connection
        x2 = self.norm2(F.leaky_relu(self.conv2(x1, edge_index), 0.2) + x1)

        # Third graph convolution
        x3 = self.norm3(F.leaky_relu(self.conv3(x2, edge_index), 0.2))

        # VAE encoding
        mu = self.fc_mu(x3)
        logvar = self.fc_logvar(x3)

        return x3, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, edge_index, edge_attr, h=None):
        # Encode
        x_encoded, mu, logvar = self.encode(x, edge_index, edge_attr)

        # Auxiliary prediction directly from encoded features
        aux_out = self.aux_classifier(x_encoded)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode for reconstruction
        x_recon = self.decode(z)

        # GRU processing
        z_sequence = z.unsqueeze(1)  # Add sequence dimension

        if h is None:
            # Initialize hidden state if not provided
            h = torch.zeros(2 * 3, z.size(0), mu.size(1), device=x.device)  # 2 for bidirectional, 3 for num_layers

        gru_out, h = self.gru(z_sequence, h)

        # Final MLP prediction
        main_out = self.mlp(gru_out.squeeze(1))

        return main_out, aux_out, x_recon, h, mu, logvar

def create_graph_from_batch(features, labels, k=10, epsilon=0.5):
    # Compute cosine similarity
    sim = cosine_similarity(features)

    # Create edges based on similarity threshold and k-nearest neighbors
    edge_index = []
    edge_weights = []

    for i in range(sim.shape[0]):
        # Get indices of top-k similar nodes
        neighbors_idx = sim[i].argsort()[-k-1:-1]

        for j in neighbors_idx:
            # Add edge if similarity is above threshold
            if sim[i, j] > epsilon:
                edge_index.append([i, j])
                edge_weights.append([sim[i, j]])  # Use similarity as edge weight

    # If no edges above threshold, fall back to basic k-nn
    if len(edge_index) == 0:
        for i in range(sim.shape[0]):
            neighbors_idx = sim[i].argsort()[-k-1:-1]
            for j in neighbors_idx:
                edge_index.append([i, j])
                edge_weights.append([sim[i, j]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    x = torch.tensor(features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), torch.tensor(labels)

def compute_anomaly_scores(main_pred, aux_pred, recon_error, mu, logvar, alpha=0.6, beta=0.2, gamma=0.2):
    # Convert predictions to probabilities
    main_prob = torch.sigmoid(main_pred).squeeze()
    aux_prob = torch.sigmoid(aux_pred).squeeze()

    # Compute reconstruction probability (1 - normalized error)
    recon_prob = 1 - torch.nn.functional.normalize(recon_error, p=2, dim=0)

    # Compute KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_prob = 1 - torch.nn.functional.normalize(kl_div, p=2, dim=0)

    # Ensemble score
    ensemble_score = alpha * main_prob + beta * aux_prob + gamma * (recon_prob + kl_prob) / 2

    return ensemble_score.detach().cpu().numpy()
