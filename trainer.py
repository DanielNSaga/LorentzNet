import os
import glob
import time
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import psutil
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
from tqdm import tqdm
import contextlib

# ---------------------------
# Eksempel: Streaming-datasett for LorentzNet
# ---------------------------
class StreamingLorentzDataset(IterableDataset):
    """
    Iterer over alle shard-filer (for eksempel lagret som "shard_0.pt", "shard_1.pt", ...)
    Hver shard inneholder en liste med (label, p4s, nodes, atom_mask).
    """
    def __init__(self, shards_dir, pattern="shard_*.pt"):
        super().__init__()
        self.shard_files = sorted(glob.glob(os.path.join(shards_dir, pattern)))
        if not self.shard_files:
            raise RuntimeError(f"Ingen shard-filer funnet i {shards_dir} med mønster {pattern}")

    def __iter__(self):
        for shard in self.shard_files:
            # Last inn én shard om gangen – spesifiser weights_only=False for å laste hele objektet
            samples = torch.load(shard, weights_only=False)
            for sample in samples:
                yield sample

# ---------------------------
# Collate-funksjon
# ---------------------------
def collate_fn(data):
    # data er en liste med tuples: (label, p4s, nodes, atom_mask)
    labels, p4s, nodes, atom_masks = zip(*data)
    labels = torch.stack(labels)         # (B,)
    p4s = torch.stack(p4s)                # (B, n_nodes, 4)
    nodes = torch.stack(nodes)            # (B, n_nodes, node_features)
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
# Modell-definisjoner (LorentzNet, LGEB, osv.)
# ---------------------------
import torch.nn as nn

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

class LGEB(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_node_attr=0,
                 dropout=0., c_weight=1.0, last_layer=False):
        super(LGEB, self).__init__()
        self.c_weight = c_weight
        n_edge_attr = 2

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
    def __init__(self, n_scalar, n_hidden, n_class=2, n_layers=6, c_weight=1e-3, dropout=0.):
        super(LorentzNet, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.LGEBs = nn.ModuleList([
            LGEB(self.n_hidden, self.n_hidden, self.n_hidden, n_node_attr=n_scalar,
                 dropout=dropout, c_weight=c_weight, last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])
        self.graph_dec = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.n_hidden, n_class)
        )

    def forward(self, scalars, x, edges, node_mask, edge_mask, n_nodes):
        B = scalars.shape[0]
        # Utvid h til alle noder: (B, n_hidden) -> (B*n_nodes, n_hidden)
        h = self.embedding(scalars)
        h = h.repeat_interleave(n_nodes, dim=0)
        # Flate ut x: (B, n_nodes, 4) -> (B*n_nodes, 4)
        x = x.view(B * n_nodes, -1)
        # Utvid node_attr tilsvarende
        node_attr = scalars.repeat_interleave(n_nodes, dim=0)
        for i in range(self.n_layers):
            h, x, _ = self.LGEBs[i](h, x, edges, node_attr=node_attr)
        # Masker h med node_mask
        node_mask = node_mask.view(B * n_nodes, 1).to(h.dtype)
        h = h * node_mask
        # Gjenoppbygg batch-dimensjonen
        h = h.view(B, n_nodes, self.n_hidden)
        h = torch.mean(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)

# ---------------------------
# Hyperparametere og Args
# ---------------------------
class Args:
    def __init__(self):
        # Modellparametere for LorentzNet
        self.n_scalar = 2      # For eksempel invariant masse og ladning
        self.n_hidden = 64
        self.n_class = 10      # Antall klasser
        self.n_layers = 6
        self.c_weight = 1e-3
        self.dropout = 0.0

        # Treningsparametere
        self.epochs = 10
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mappen der shard-filene ligger
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
        scaler = torch.cuda.amp.GradScaler()
    else:
        # Dummy scaler for CPU
        class DummyScaler:
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        scaler = DummyScaler()

    # Bruk streaming-datasettet
    stream_dataset = StreamingLorentzDataset(args.shards_dir)
    num_shards = len(glob.glob(os.path.join(args.shards_dir, 'shard_*.pt')))
    print(f"Streaming-datasettet inneholder {num_shards} shards.")

    train_loader = DataLoader(stream_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(stream_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(stream_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    model = LorentzNet(n_scalar=args.n_scalar, n_hidden=args.n_hidden, n_class=args.n_class,
                       n_layers=args.n_layers, c_weight=args.c_weight, dropout=args.dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    TRAIN_BATCHES_TOTAL = 12500
    VAL_BATCHES_TOTAL   = 1562
    TEST_BATCHES_TOTAL  = 1562

    def train_epoch(model, optimizer, loader, epoch):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True, total=TRAIN_BATCHES_TOTAL)
        for i, batch in enumerate(pbar):
            labels, p4s, nodes, atom_mask, edge_mask, edges = batch
            labels = labels.to(DEVICE)
            p4s = p4s.to(DEVICE)
            nodes = nodes.to(DEVICE)
            atom_mask = atom_mask.to(DEVICE)
            edge_mask = edge_mask.to(DEVICE)
            scalars = nodes.mean(dim=1)
            n_nodes = p4s.shape[1]
            # Cast p4s til fp16 kun på CUDA
            if DEVICE.type == "cuda":
                p4s = p4s.to(torch.float16)
            else:
                p4s = p4s.to(torch.float32)
            optimizer.zero_grad()
            with (torch.amp.autocast(device_type="cuda") if DEVICE.type=="cuda" else contextlib.nullcontext()):
                outputs = model(scalars, p4s, edges, atom_mask, edge_mask, n_nodes)
                targets = labels.long()
                loss = F.cross_entropy(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_samples += bs
            cur_loss = total_loss / total_samples
            cur_acc  = total_correct / total_samples
            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")
            if i + 1 >= TRAIN_BATCHES_TOTAL:
                break
        return total_loss / total_samples, total_correct / total_samples

    def validate_epoch(model, loader, epoch_str="Val", is_val=True):
        model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        total_steps = VAL_BATCHES_TOTAL if is_val else TEST_BATCHES_TOTAL
        pbar = tqdm(loader, desc=f"{epoch_str}", leave=True, total=total_steps)
        with torch.no_grad():
            for i, batch in enumerate(pbar):
                labels, p4s, nodes, atom_mask, edge_mask, edges = batch
                labels = labels.to(DEVICE)
                p4s = p4s.to(DEVICE)
                nodes = nodes.to(DEVICE)
                atom_mask = atom_mask.to(DEVICE)
                edge_mask = edge_mask.to(DEVICE)
                scalars = nodes.mean(dim=1)
                n_nodes = p4s.shape[1]
                if DEVICE.type == "cuda":
                    p4s = p4s.to(torch.float16)
                else:
                    p4s = p4s.to(torch.float32)
                with (torch.amp.autocast(device_type="cuda") if DEVICE.type=="cuda" else contextlib.nullcontext()):
                    out = model(scalars, p4s, edges, atom_mask, edge_mask, n_nodes)
                targets = labels.long()
                loss = F.cross_entropy(out, targets)
                preds = out.argmax(dim=1)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                bs = targets.size(0)
                total_loss    += loss.item() * bs
                total_correct += (preds == targets).sum().item()
                total_samples += bs
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                all_probs.extend(probs)
                cur_loss = total_loss / total_samples
                cur_acc  = total_correct / total_samples
                pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")
                if i + 1 >= total_steps:
                    break
        roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        idx = np.where(tpr >= 0.90)[0][0] if np.any(tpr >= 0.90) else -1
        fpr90 = fpr[idx] if idx >= 0 and idx < len(fpr) else fpr[-1]
        return total_loss / total_samples, total_correct / total_samples, roc_auc, fpr90

    @torch.no_grad()
    def measure_inference_latency(model, loader, num_samples=1000):
        model.eval()
        times = []
        for count, batch in enumerate(loader):
            if count >= num_samples:
                break
            labels, p4s, nodes, atom_mask, edge_mask, edges = batch
            labels = labels.to(DEVICE)
            p4s = p4s.to(DEVICE)
            nodes = nodes.to(DEVICE)
            atom_mask = atom_mask.to(DEVICE)
            edge_mask = edge_mask.to(DEVICE)
            scalars = nodes.mean(dim=1)
            n_nodes = p4s.shape[1]
            if DEVICE.type == "cuda":
                p4s = p4s.to(torch.float16)
            else:
                p4s = p4s.to(torch.float32)
            with (torch.amp.autocast(device_type="cuda") if DEVICE.type=="cuda" else contextlib.nullcontext()):
                start = time.time()
                _ = model(scalars, p4s, edges, atom_mask, edge_mask, n_nodes)
                end = time.time()
            times.append(end - start)
        return np.mean(times) * 1000.0 if times else 0.0

    @torch.no_grad()
    def permutation_feature_importance(model, loader, input_dim, n_perm=3):
        model.eval()
        baseline_acc = validate_epoch(model, loader, epoch_str="Baseline", is_val=False)[1]
        importances = np.zeros(input_dim, dtype=np.float32)
        for feat_idx in range(input_dim):
            results = []
            for _ in range(n_perm):
                for batch in loader:
                    labels, p4s, nodes, atom_mask, edge_mask, edges = batch
                    labels = labels.to(DEVICE)
                    p4s = p4s.to(DEVICE)
                    nodes = nodes.to(DEVICE)
                    atom_mask = atom_mask.to(DEVICE)
                    edge_mask = edge_mask.to(DEVICE)
                    scalars = nodes.mean(dim=1)
                    n_nodes = p4s.shape[1]
                    original_vals = nodes[:, :, feat_idx].clone()
                    permuted_vals = original_vals[torch.randperm(original_vals.size(0))]
                    nodes[:, :, feat_idx] = permuted_vals
                    perm_acc = validate_epoch(model, [(labels, p4s, nodes, atom_mask, edge_mask, edges)],
                                              epoch_str="PermTest", is_val=False)[1]
                    results.append(baseline_acc - perm_acc)
                    nodes[:, :, feat_idx] = original_vals  # reset
            importances[feat_idx] = float(np.mean(results))
            print(f"Feature {feat_idx:2d} => importance={importances[feat_idx]:.4f}")
        return importances

    # ---------------------------
    # Treningsløkken
    # ---------------------------
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, epoch)
        val_loss, val_acc, val_roc, val_fpr90 = validate_epoch(model, val_loader, epoch_str=f"Val epoch={epoch}", is_val=True)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f} ROC={val_roc:.4f} FPR90={val_fpr90:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict()}, "best_lorentznet.pt")
            print(f"🔹 Lagret checkpoint ved epoch {epoch}")

    print(f"📥 Laster beste checkpoint fra epoch {best_epoch}")
    model.load_state_dict(torch.load("best_lorentznet.pt")["model_state_dict"])
    model.to(DEVICE)

    print("📊 Evaluering på test-sett:")
    test_loss, test_acc, test_roc, test_fpr90 = validate_epoch(model, test_loader, epoch_str="Test", is_val=False)
    print(f"Test => Loss={test_loss:.4f}, Acc={test_acc:.4f}, ROC={test_roc:.4f}, FPR90={test_fpr90:.4f}")

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
            p4s = p4s.to(DEVICE)
            nodes = nodes.to(DEVICE)
            atom_mask = atom_mask.to(DEVICE)
            edge_mask = edge_mask.to(DEVICE)
            scalars = nodes.mean(dim=1)
            n_nodes = p4s.shape[1]
            if DEVICE.type == "cuda":
                p4s = p4s.to(torch.float16)
            else:
                p4s = p4s.to(torch.float32)
            with (torch.amp.autocast(device_type="cuda") if DEVICE.type=="cuda" else contextlib.nullcontext()):
                out = model(scalars, p4s, edges, atom_mask, edge_mask, n_nodes)
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

    print("📊 Kjører permutasjonsbasert feature importance:")
    importances = permutation_feature_importance(model, test_loader, input_dim=args.n_scalar, n_perm=3)
    np.savetxt("permutation_importances_lorentznet.txt", importances, fmt="%.4f")
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(importances)), importances)
    plt.xlabel("Feature index")
    plt.ylabel("Importance (Δ acc)")
    plt.title("Permutation-based Feature Importances")
    plt.savefig("permutation_importances_lorentznet.png")
    plt.show()

    print("✅ Fullført trening!")
