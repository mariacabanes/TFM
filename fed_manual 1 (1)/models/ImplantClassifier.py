import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torch_geometric.nn import GCNConv, global_mean_pool

class ImplantClassifier(nn.Module):
    def __init__(self, cnn_out=128, gnn_out=64, scalar_out=32, n_brands=5, n_diameters=8):
        super().__init__()
        self.cnn_backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        self.cnn_backbone.classifier = nn.Identity()
        self.cnn_fc = nn.Linear(1408, cnn_out)
        self.gcn1 = GCNConv(2, 32)
        self.gcn2 = GCNConv(32, gnn_out)
        self.scalar_fc = nn.Linear(7, scalar_out)
        self.fc = nn.Sequential(nn.Linear(cnn_out + gnn_out + scalar_out, 128), nn.ReLU(), nn.Dropout(0.3))
        self.brand_head = nn.Linear(128, n_brands)
        self.diameter_head = nn.Linear(128, n_diameters)

    def forward(self, image, node_feats, edge_index, batch, scalar_feat):
        img_feat = self.cnn_fc(self.cnn_backbone(image))
        x = torch.relu(self.gcn1(node_feats, edge_index))
        x = self.gcn2(x, edge_index)
        gnn_feat = global_mean_pool(x, batch)
        scalar_emb = self.scalar_fc(scalar_feat)
        combined = torch.cat([img_feat, gnn_feat, scalar_emb], dim=1)
        out = self.fc(combined)
        return self.brand_head(out), self.diameter_head(out)