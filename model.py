import torch
from torch import nn


# --- Støttefunksjoner for LorentzNet-modellen ---
def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def normsq4(p):
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p, q):
    psq = p * q
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def psi(p):
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


# --- LGEB-laget ---
class LGEB(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_node_attr=0,
                 dropout=0., c_weight=1.0, last_layer=False):
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2  # dimensjoner for Minkowski-norm og indreprodukt

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU()
        )

        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi_x = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            layer
        )

        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )

        self.last_layer = last_layer
        if last_layer:
            del self.phi_x

    def m_model(self, hi, hj, norms, dots):
        out = torch.cat([hi, hj, norms, dots], dim=1)
        out = self.phi_e(out)
        w = self.phi_m(out)
        return out * w

    def h_model(self, h, edges, m, node_attr):
        i, _ = edges  # vi summerer over mottakere (i)
        agg = unsorted_segment_sum(m, i, num_segments=h.size(0))
        agg = torch.cat([h, agg, node_attr], dim=1)
        return h + self.phi_h(agg)

    def x_model(self, x, edges, x_diff, m):
        i, _ = edges
        trans = x_diff * self.phi_x(m)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, i, num_segments=x.size(0))
        return x + agg * self.c_weight

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = normsq4(x_diff).unsqueeze(1)
        dots = dotsq4(x[i], x[j]).unsqueeze(1)
        return psi(norms), psi(dots), x_diff

    def forward(self, h, x, edges, node_attr=None):
        norms, dots, x_diff = self.minkowski_feats(edges, x)
        m = self.m_model(h[edges[0]], h[edges[1]], norms, dots)
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
        h = self.h_model(h, edges, m, node_attr)
        return h, x, m


# --- Hovedmodellen LorentzNet ---
class LorentzNet(nn.Module):
    """
    LorentzNet-modell.

    Args:
        n_scalar (int): antall scalar input-funksjoner (f.eks. jet_pt).
        n_hidden (int): dimensjon på latentrommet.
        n_class (int): antall utgangsklasser.
        n_layers (int): antall LGEB-lag.
        c_weight (float): parameter for x_model.
        dropout (float): dropout-rate.
    """

    def __init__(self, n_scalar, n_hidden, n_class=2, n_layers=6, c_weight=1e-3, dropout=0.):
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList([
            LGEB(n_input=self.n_hidden, n_output=self.n_hidden, n_hidden=self.n_hidden,
                 n_node_attr=n_scalar, dropout=dropout, c_weight=c_weight, last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])
        self.graph_dec = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_class)
        )

    def forward(self, scalars, x, edges, node_mask, edge_mask, n_nodes):
        # scalars: (B, n_scalar)
        B = scalars.shape[0]
        # Repetér scalars for hver node: (B, n_scalar) → (B, N, n_scalar) → (B*N, n_scalar)
        scalars_rep = scalars.unsqueeze(1).repeat(1, n_nodes, 1).view(B * n_nodes, -1)
        h = self.embedding(scalars_rep)  # h: (B*N, n_hidden)
        for lgeb in self.LGEBs:
            h, x, _ = lgeb(h, x, edges, node_attr=scalars_rep)
        h = h.view(B, n_nodes, self.n_hidden)
        h = torch.mean(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


# --- Dataadapter for å mappe batch til modellinput ---
class LorentzNetDataAdapter:
    """
    Adapter som tar en batch (dictionary) fra DataLoader og konverterer
    til input til LorentzNet: scalars, x (Pmu), edges, node_mask, edge_mask, n_nodes.
    """

    def __call__(self, batch):
        # Sørg for at tensorene er float
        scalars = batch["scalars"].float()  # (B, n_scalar)
        x = batch["Pmu"].float()  # (B, N, 4)
        edges = batch["edges"]  # liste [edge_rows, edge_cols]
        node_mask = batch["atom_mask"].float()  # (B, N)
        edge_mask = batch["edge_mask"].float()  # (B, N, N)
        n_nodes = x.size(1)
        return scalars, x, edges, node_mask, edge_mask, n_nodes


