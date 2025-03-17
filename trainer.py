import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm  # 🔥 Progressbar
from dataset import LorentzNetH5Dataset, collate_fn
from model import LorentzNet

# --- Adapter som flater ut 4-momenta og beholder scalar input som (B, n_scalar) ---
class FlatteningAdapter:
    def __call__(self, batch):
        scalars = batch["scalars"].float()
        x = batch["Pmu"].float()
        B, N, _ = x.shape
        x_flat = x.view(B * N, 4)
        node_mask = batch["atom_mask"].float().view(B * N)
        return scalars, x_flat, batch["edges"], node_mask, batch["edge_mask"], N

# --- Dataset-wrapper som legger til jet_label basert på filtilhørighet ---
class LabeledLorentzNetDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, file_to_label):
        self.base_dataset = base_dataset
        self.file_to_label = file_to_label
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        file_path, _ = self.base_dataset.file_indices[idx]
        sample['jet_label'] = torch.tensor(self.file_to_label[file_path], dtype=torch.long)
        return sample

# --- Funksjon for balansert splitting ---
def create_balanced_indices(dataset, target_counts):
    file_to_indices = {}
    for i, (f, _) in enumerate(dataset.file_indices):
        file_to_indices.setdefault(f, []).append(i)
    balanced_indices = {}
    for f, indices in file_to_indices.items():
        random.shuffle(indices)
        balanced_indices[f] = {
            'train': indices[:target_counts['train']],
            'val': indices[target_counts['train']:target_counts['train'] + target_counts['val']],
            'test': indices[target_counts['train'] + target_counts['val']:
                              target_counts['train'] + target_counts['val'] + target_counts['test']]
        }
    train_indices, val_indices, test_indices = [], [], []
    for f in balanced_indices:
        train_indices.extend(balanced_indices[f]['train'])
        val_indices.extend(balanced_indices[f]['val'])
        test_indices.extend(balanced_indices[f]['test'])
    return train_indices, val_indices, test_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='Data/')
    parser.add_argument('--train_total', type=int, default=200000)
    parser.add_argument('--val_total', type=int, default=50000)
    parser.add_argument('--test_total', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # Sett frø for reproducerbarhet
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Last inn datasettet (alle h5-filer)
    full_dataset = LorentzNetH5Dataset(args.data_folder)
    unique_files = sorted(list({f for (f, _) in full_dataset.file_indices}))
    file_to_label = {f: i for i, f in enumerate(unique_files)}
    num_classes = len(unique_files)

    # Beregn antall eksempler per fil for hvert split
    train_per_file = args.train_total // num_classes
    val_per_file = args.val_total // num_classes
    test_per_file = args.test_total // num_classes
    target_counts = {'train': train_per_file, 'val': val_per_file, 'test': test_per_file}

    # Wrapp datasettet slik at hvert sample får jet_label
    labeled_dataset = LabeledLorentzNetDataset(full_dataset, file_to_label)
    train_indices, val_indices, test_indices = create_balanced_indices(full_dataset, target_counts)

    from torch.utils.data import Subset
    train_dataset = Subset(labeled_dataset, train_indices)
    val_dataset = Subset(labeled_dataset, val_indices)
    test_dataset = Subset(labeled_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # Instansier modellen
    model = LorentzNet(n_scalar=1, n_hidden=64, n_class=num_classes, n_layers=6, c_weight=1e-3, dropout=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    adapter = FlatteningAdapter()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Treningsløkke med progressbar
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")  # 🚀 Progressbar
        for batch in train_pbar:
            optimizer.zero_grad()
            scalars, x_flat, edges, node_mask, edge_mask, n_nodes = adapter(batch)
            scalars, x_flat, node_mask, edge_mask = scalars.to(device), x_flat.to(device), node_mask.to(device), edge_mask.to(device)
            edges = [e.to(device) for e in edges]
            outputs = model(scalars, x_flat, edges, node_mask, edge_mask, n_nodes)
            targets = batch["jet_label"].to(device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())  # 🔄 Oppdaterer progressbaren

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f}")

        # Valideringsfase med progressbar
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]")  # 🚀 Progressbar
        with torch.no_grad():
            for batch in val_pbar:
                scalars, x_flat, edges, node_mask, edge_mask, n_nodes = adapter(batch)
                scalars, x_flat, node_mask, edge_mask = scalars.to(device), x_flat.to(device), node_mask.to(device), edge_mask.to(device)
                edges = [e.to(device) for e in edges]
                outputs = model(scalars, x_flat, edges, node_mask, edge_mask, n_nodes)
                targets = batch["jet_label"].to(device)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                val_pbar.set_postfix(loss=loss.item())  # 🔄 Oppdaterer progressbaren

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

if __name__ == '__main__':
    main()
