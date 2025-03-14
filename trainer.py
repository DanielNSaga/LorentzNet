import os
import glob
import time
import random
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import psutil
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


# ---------------------------
# Streaming-datasett for LorentzNet
# ---------------------------
class StreamingLorentzDataset(IterableDataset):
    """
    Iterer over en liste med shard-filer.
    Hver shard er en .pt-fil som inneholder en liste med (label, p4s, nodes, atom_mask).
    Under iterasjon stokkes rekkefølgen på shardene og sample-listen i hver shard.
    """

    def __init__(self, shard_files):
        super().__init__()
        self.shard_files = sorted(shard_files)
        if not self.shard_files:
            raise RuntimeError("Ingen shard-filer funnet!")

    def __iter__(self):
        shards = self.shard_files.copy()
        random.shuffle(shards)
        for shard in shards:
            samples = torch.load(shard, weights_only=False)
            random.shuffle(samples)
            for sample in samples:
                yield sample


# ---------------------------
# Funksjon for å splitte shard-filer
# ---------------------------
def split_shards(shards_dir, pattern="shard_*.pt", train_frac=0.8, val_frac=0.1):
    shard_files = sorted(glob.glob(os.path.join(shards_dir, pattern)))
    n = len(shard_files)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_shards = shard_files[:n_train]
    val_shards = shard_files[n_train:n_train + n_val]
    test_shards = shard_files[n_train + n_val:]
    logging.info(
        f"Splitter {n} shards til: {len(train_shards)} trenings-, {len(val_shards)} validerings-, {len(test_shards)} test-shards.")
    return train_shards, val_shards, test_shards


# ---------------------------
# Collate-funksjon
# ---------------------------
def collate_fn(data):
    # data er en liste med tuples: (label, p4s, nodes, atom_mask)
    labels, p4s, nodes, atom_masks = zip(*data)
    labels = torch.stack(labels)  # (B,)
    p4s = torch.stack(p4s)  # (B, n_nodes, 4)
    nodes = torch.stack(nodes)  # (B, n_nodes, 4)
    atom_masks = torch.stack(atom_masks)  # (B, n_nodes)

    batch_size, n_nodes, _ = p4s.shape
    # Bygg edge_mask: True for ekte partikler (uten padding)
    edge_mask = atom_masks.unsqueeze(1) * atom_masks.unsqueeze(2)
    # Fjern diagonal (selv-koblinger)
    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool).unsqueeze(0)
    edge_mask = edge_mask * diag_mask

    # Bygg kantindeks (edges) fra edge_mask
    rows, cols = [], []
    from scipy.sparse import coo_matrix
    for batch_idx in range(batch_size):
        offset = batch_idx * n_nodes
        x = coo_matrix(edge_mask[batch_idx].numpy())
        rows.append(offset + x.row)
        cols.append(offset + x.col)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    edges = [torch.LongTensor(rows), torch.LongTensor(cols)]

    return labels, p4s, nodes, atom_masks, edge_mask, edges


# ---------------------------
# Modell-definisjoner
# ---------------------------
import torch.nn as nn


def normsq4(p):
    """Minkowski-norm kvadrat: ||p||² = p₀² - p₁² - p₂² - p₃²"""
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p, q):
    """Minkowski indre produkt: <p,q> = p₀q₀ - p₁q₁ - p₂q₂ - p₃q₃"""
    psq = p * q
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def psi(p):
    """Feature-transformasjon for Minkowski-verdier: ψ(p) = sign(p) * log(|p| + 1)"""
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


class LGEB(nn.Module):
    """Lorentz Group Equivariant Block (LGEB)"""

    def __init__(self, n_input, n_output, n_hidden, n_node_attr=0,
                 dropout=0., c_weight=1.0, last_layer=False):
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2  # Minkowski-norm & indre produkt

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
        i, j = edges
        agg = unsorted_segment_sum(m, i, num_segments=h.size(0))
        agg = torch.cat([h, agg, node_attr], dim=1)
        return h + self.phi_h(agg)

    def x_model(self, x, edges, x_diff, m):
        i, j = edges
        trans = x_diff * self.phi_x(m)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, i, num_segments=x.size(0))
        return x + agg * self.c_weight

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = normsq4(x_diff).unsqueeze(1)
        dots = dotsq4(x[i], x[j]).unsqueeze(1)
        norms, dots = psi(norms), psi(dots)
        return norms, dots, x_diff

    def forward(self, h, x, edges, node_attr=None):
        i, j = edges
        norms, dots, x_diff = self.minkowski_feats(edges, x)
        m = self.m_model(h[i], h[j], norms, dots)
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
        h = self.h_model(h, edges, m, node_attr)
        return h, x, m


class LorentzNet(nn.Module):
    def __init__(self, n_scalar, n_hidden, n_class, n_layers=6, c_weight=1e-3, dropout=0.):
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        # Forventer at input er 4-dimensjonal (fire-vektor)
        self.embedding = nn.Linear(4, n_hidden)

        self.LGEBs = nn.ModuleList([
            LGEB(self.n_hidden, self.n_hidden, self.n_hidden,
                 n_node_attr=4, dropout=dropout, c_weight=c_weight, last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])

        self.graph_dec = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.n_hidden, n_class)
        )

    def forward(self, p4, x, edges, node_mask, edge_mask, n_nodes):
        # p4 har formen (B, n_nodes, 4)
        B = p4.shape[0]
        # Flate ut til (B*n_nodes, 4)
        p4_flat = p4.view(B * n_nodes, 4)
        # Bruk p4_flat for embedding og som initial x
        h = self.embedding(p4_flat)
        x = p4_flat.clone()
        # Bruk også p4_flat som node_attr i LGEB-blokkene
        for i in range(self.n_layers):
            h, x, _ = self.LGEBs[i](h, x, edges, node_attr=p4_flat)
        # Reshape h til (B, n_nodes, hidden)
        h = h.view(B, n_nodes, self.n_hidden)
        # Tilpass node_mask (B, n_nodes) til (B, n_nodes, 1) før maskering
        node_mask = node_mask.view(B, n_nodes, 1)
        h = h * node_mask
        # Aggreger nodene (for eksempel ved gjennomsnitt)
        h = torch.mean(h, dim=1)
        return self.graph_dec(h).squeeze(1)



# ---------------------------
# Hjelpefunksjoner for segment-sum og mean
# ---------------------------
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


# ---------------------------
# Hyperparametere og Args
# ---------------------------
class Args:
    def __init__(self):
        self.n_scalar = 2  # f.eks. invariant masse og ladning
        self.n_hidden = 64
        self.n_class = 10  # Binær klassifisering
        self.n_layers = 6
        self.c_weight = 1e-3
        self.dropout = 0.0

        self.epochs = 10
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.shards_dir = "./processed_dataset/shards"


args = Args()

# ---------------------------
# Hovedtreningskode (i main-blokk)
# ---------------------------
if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('fork', force=True)

    DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Splitte shard-filene i separate sett for trening, validering og testing:
    train_shards, val_shards, test_shards = split_shards(args.shards_dir)

    train_dataset = StreamingLorentzDataset(train_shards)
    val_dataset = StreamingLorentzDataset(val_shards)
    test_dataset = StreamingLorentzDataset(test_shards)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn,
                             pin_memory=True)

    model = LorentzNet(n_scalar=args.n_scalar, n_hidden=args.n_hidden, n_class=args.n_class,
                       n_layers=args.n_layers, c_weight=args.c_weight, dropout=args.dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    TRAIN_BATCHES_TOTAL = 12500
    VAL_BATCHES_TOTAL = 1562
    TEST_BATCHES_TOTAL = 1562


    def train_epoch(model, optimizer, loader, epoch):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True, total=TRAIN_BATCHES_TOTAL)
        for i, batch in enumerate(pbar):
            labels, p4s, nodes, atom_mask, edge_mask, edges = batch
            labels = labels.to(DEVICE)
            p4s = p4s.to(DEVICE).to(torch.float32)
            nodes = nodes.to(DEVICE).to(torch.float32)
            atom_mask = atom_mask.to(DEVICE)
            edge_mask = edge_mask.to(DEVICE)
            n_nodes = p4s.shape[1]
            optimizer.zero_grad()
            # Bruk p4s direkte som input til modellen
            outputs = model(p4s, p4s, edges, atom_mask, edge_mask, n_nodes)
            targets = labels.long()
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += bs
            cur_loss = total_loss / total_samples
            cur_acc = total_correct / total_samples
            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")
            if i + 1 >= TRAIN_BATCHES_TOTAL:
                break
        return total_loss / total_samples, total_correct / total_samples


    def validate_epoch(model, loader, epoch_str="Val"):
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(loader, desc=epoch_str, leave=True, total=VAL_BATCHES_TOTAL)
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                labels, p4s, nodes, atom_mask, edge_mask, edges = batch
                labels = labels.to(DEVICE)
                p4s = p4s.to(DEVICE).to(torch.float32)
                nodes = nodes.to(DEVICE).to(torch.float32)
                atom_mask = atom_mask.to(DEVICE)
                edge_mask = edge_mask.to(DEVICE)
                n_nodes = p4s.shape[1]
                out = model(p4s, p4s, edges, atom_mask, edge_mask, n_nodes)
                targets = labels.long()
                loss = F.cross_entropy(out, targets)
                preds = out.argmax(dim=1)
                bs = targets.size(0)
                total_loss += loss.item() * bs
                total_correct += (preds == targets).sum().item()
                total_samples += bs
                cur_loss = total_loss / total_samples
                cur_acc = total_correct / total_samples
                pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")
                if i + 1 >= VAL_BATCHES_TOTAL:
                    break
        return total_loss / total_samples, total_correct / total_samples


    @torch.no_grad()
    def measure_inference_latency(model, loader, num_samples=1000):
        model.eval()
        times = []
        for count, batch in enumerate(loader):
            if count >= num_samples:
                break
            labels, p4s, nodes, atom_mask, edge_mask, edges = batch
            labels = labels.to(DEVICE)
            p4s = p4s.to(DEVICE).to(torch.float32)
            nodes = nodes.to(DEVICE).to(torch.float32)
            atom_mask = atom_mask.to(DEVICE)
            edge_mask = edge_mask.to(DEVICE)
            n_nodes = p4s.shape[1]
            start = time.time()
            _ = model(p4s, p4s, edges, atom_mask, edge_mask, n_nodes)
            end = time.time()
            times.append(end - start)
        return np.mean(times) * 1000.0 if times else 0.0


    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, epoch)
        val_loss, val_acc = validate_epoch(model, val_loader, epoch_str=f"Val epoch={epoch}")
        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict()}, "best_lorentznet.pt")
            print(f"🔹 Lagret checkpoint ved epoch {epoch}")

    print(f"📥 Laster beste checkpoint fra epoch {best_epoch}")
    model.load_state_dict(torch.load("best_lorentznet.pt")["model_state_dict"])
    model.to(DEVICE)

    print("📊 Evaluering på test-sett:")
    test_loss, test_acc = validate_epoch(model, test_loader, epoch_str="Test")
    print(f"Test => Loss={test_loss:.4f}, Acc={test_acc:.4f}")

    latency = measure_inference_latency(model, test_loader)
    print(f"🕒 Inference-latency per batch ~ {latency:.3f} ms")

    memory_usage = psutil.virtual_memory().used / 1e9
    print(f"💾 Memory usage ~ {memory_usage:.2f} GB")

    # Confusion Matrix
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels, p4s, nodes, atom_mask, edge_mask, edges = batch
            labels = labels.to(DEVICE)
            p4s = p4s.to(DEVICE).to(torch.float32)
            nodes = nodes.to(DEVICE).to(torch.float32)
            atom_mask = atom_mask.to(DEVICE)
            edge_mask = edge_mask.to(DEVICE)
            n_nodes = p4s.shape[1]
            out = model(p4s, p4s, edges, atom_mask, edge_mask, n_nodes)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_lorentznet.png")
    plt.show()

    print("📊 Klassifikasjonsrapport:")
    print(classification_report(all_labels, all_preds, digits=4))

    print("✅ Fullført trening!")
