import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Hjelpefunksjoner for batch-stacking og padding (for collate_fn)
def batch_stack_general(props):
    if isinstance(props[0], (int, float)):
        return torch.tensor(props)
    if isinstance(props[0], np.ndarray):
        props = [torch.from_numpy(prop) for prop in props]
    shapes = [p.shape for p in props]
    if all(shapes[0] == s for s in shapes):
        return torch.stack(props)
    elif all(shapes[0][1:] == s[1:] for s in shapes):
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
    else:
        raise ValueError("Ulike dimensjoner ved batch_stack_general!")


def get_adj_matrix(n_nodes, batch_size, edge_mask):
    from scipy.sparse import coo_matrix
    rows, cols = [], []
    for b in range(batch_size):
        offset = b * n_nodes
        mat = coo_matrix(edge_mask[b].cpu().numpy())
        rows.append(offset + mat.row)
        cols.append(offset + mat.col)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    return [torch.LongTensor(rows), torch.LongTensor(cols)]


def collate_fn(data, scale=1.):
    # Stack alle felter til batch
    batch = {key: batch_stack_general([d[key] for d in data]) for key in data[0].keys()}

    # Sjekk batch-størrelse
    batch_size = batch['Pmu'].shape[0]
    if batch_size == 0:
        raise ValueError("[ERROR] Tom batch! Noe er galt med dataene.")

    # Sjekk `Pmu` format og NaN-verdier
    if batch['Pmu'].dim() != 3 or batch['Pmu'].shape[-1] != 4:
        raise ValueError(f"[ERROR] `Pmu` har feil dimensjon: {batch['Pmu'].shape}, forventet (batch, N_particles, 4)")
    if torch.isnan(batch['Pmu']).any():
        raise ValueError("[ERROR] `Pmu` inneholder NaN-verdier!")

    # Sjekk `label` format
    batch['label'] = batch['label'].to(torch.bool)
    if batch['label'].shape != batch['Pmu'].shape[:-1]:
        raise ValueError(
            f"[ERROR] `label` har feil dimensjon: {batch['label'].shape}, forventet {batch['Pmu'].shape[:-1]}")

    # Maskering for å fjerne padding-elementer
    atom_mask = batch['Pmu'][..., 0] != 0.
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask = edge_mask * diag_mask
    batch['atom_mask'] = atom_mask.to(torch.bool)
    batch['edge_mask'] = edge_mask.to(torch.bool)

    # Lag kantliste for grafstrukturen
    batch_size, n_nodes, _ = batch['Pmu'].size()
    batch['edges'] = get_adj_matrix(n_nodes, batch_size, batch['edge_mask'])

    return batch


# Dataset-klasse med strømming fra disk
class LorentzNetH5Dataset(Dataset):
    def __init__(self, folder="Data/"):
        self.files = glob.glob(os.path.join(folder, "*.h5"))
        if not self.files:
            raise FileNotFoundError(f"[ERROR] Ingen HDF5-filer funnet i {folder}")

        # Tell totalt antall events
        self.file_indices = []
        self.total_events = 0
        for f in self.files:
            with h5py.File(f, 'r') as h:
                n_events = h["Nobj"].shape[0]
                self.file_indices.extend([(f, i) for i in range(n_events)])
                self.total_events += n_events

    def __len__(self):
        return self.total_events

    def __getitem__(self, idx):
        file_path, event_idx = self.file_indices[idx]
        with h5py.File(file_path, 'r') as h:
            sample = {
                "Nobj": h["Nobj"][event_idx],
                "Pmu": h["Pmu"][event_idx],
                "truth_Pmu": h["truth_Pmu"][event_idx],
                "is_signal": h["is_signal"][event_idx],
                "jet_pt": h["jet_pt"][event_idx],
                "label": h["label"][event_idx],
                "mass": h["mass"][event_idx]
            }
            # Bruk jet_pt som scalar-feature
            sample["scalars"] = np.array([h["jet_pt"][event_idx]])

        # Konverter til tensorer og sjekk for NaN
        sample_tensors = {k: torch.tensor(v) for k, v in sample.items()}
        for key, tensor in sample_tensors.items():
            if torch.isnan(tensor).any():
                raise ValueError(f"[ERROR] `{key}` inneholder NaN-verdier i fil {file_path}, event {event_idx}")

        return sample_tensors


