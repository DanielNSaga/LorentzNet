#!/usr/bin/env python
import os, glob, time
import numpy as np
import h5py
import uproot
from numba import jit


##########################################
# Lorentz-invariante funksjoner (dot, pt) #
##########################################

@jit
def dot(p1, p2):
    return p1[0] * p2[0] - np.dot(p1[1:], p2[1:])


@jit
def dots(p1s, p2s):
    return np.array([dot(p1s[i], p2s[i]) for i in range(p1s.shape[0])])


@jit
def masses(p1):
    return np.sqrt(np.maximum(0., dots(p1, p1)))


@jit
def dots_matrix_multi(p1s, p2s):
    a = p1s.shape[0]
    b = p2s.shape[0]
    return np.array([[dot(p1s[i], p2s[j]) for j in range(b)] for i in range(a)])


@jit
def dots_matrix_single(p1s):
    a = p1s.shape[0]
    matrix = np.zeros((a, a), dtype=np.dtype('f8'))
    for i in range(a):
        for j in range(i + 1):
            matrix[i, j] = dot(p1s[i], p1s[j])
            matrix[j, i] = matrix[i, j]
    return matrix


def dots_matrix(p1s, p2s):
    if np.array_equal(p1s, p2s):
        return dots_matrix_single(p1s)
    else:
        return dots_matrix_multi(p1s, p2s)


@jit
def pt(momentum):
    return np.sqrt(np.dot(momentum[1:3], momentum[1:3]))


##########################################
# Konverteringsfunksjon for ROOT-filer   #
##########################################

def convert_root_to_lorentznet(root_file, output_file, add_beams=False, dot_products=False, double_precision=True):
    precision = 'f8' if double_precision else 'f4'

    # Åpne ROOT-fil og hent treet (forutsetter at treet heter "tree")
    file = uproot.open(root_file)
    tree = file["tree"]

    # Les nødvendige grener – tilpass grenavn etter dine filer
    branches = [
        "part_energy", "part_px", "part_py", "part_pz",
        "jet_nparticles",
        "label_QCD", "label_Hbb", "label_Hcc", "label_Hgg",
        "label_H4q", "label_Hqql", "label_Zqq", "label_Wqq",
        "label_Tbqq", "label_Tbl"
    ]
    data = tree.arrays(branches, library="np")
    nentries = len(data["jet_nparticles"])
    nbeam = 2 if add_beams else 0
    nvectors_original = int(np.max(data["jet_nparticles"]))
    nvectors = nvectors_original + nbeam

    # Opprett output-dictionary med LorentzNet-formatfelter
    out = {
        "Nobj": np.zeros(nentries, dtype=np.int16),
        "Pmu": np.zeros((nentries, nvectors, 4), dtype=precision),
        "truth_Pmu": np.zeros((nentries, 4), dtype=precision),  # settes til null
        "is_signal": np.zeros(nentries, dtype=np.int16),  # her bruker vi f.eks. label_Hbb som signalflagg
        "jet_pt": np.zeros(nentries, dtype=precision),
        "label": np.zeros((nentries, nvectors), dtype=np.int16),  # partikkel-etiketter (1 for gyldige, -1 for beam)
        "mass": np.zeros((nentries, nvectors), dtype=precision),
        "jet_label": np.zeros((nentries, 10), dtype=np.int16)  # 10 jet-klasser
    }
    if dot_products:
        out["dots"] = np.zeros((nentries, nvectors, nvectors), dtype=precision)

    # Setup for beam-partikler (om de skal legges til)
    if add_beams:
        beam_mass = 0.
        beam_pz = 1.
        beam_E = np.sqrt(beam_mass ** 2 + beam_pz ** 2)
        beam_vec = np.array([beam_E, 0., 0., beam_pz], dtype=precision)
        beam_vecs = np.array([beam_vec, beam_vec], dtype=precision)
        beam_vecs[-1, -1] *= -1

    # Loop over alle events
    for i in range(nentries):
        nobj = int(data["jet_nparticles"][i])
        # Hent 4-momenta for nobj partikler: [E, px, py, pz]
        E = data["part_energy"][i][:nobj]
        px = data["part_px"][i][:nobj]
        py = data["part_py"][i][:nobj]
        pz = data["part_pz"][i][:nobj]
        Pmu_event = np.stack([E, px, py, pz], axis=1)
        out["Pmu"][i, :nobj, :] = Pmu_event
        out["Nobj"][i] = nobj + (nbeam if add_beams else 0)

        # Bruk label_Hbb som signalflagg (kan tilpasses)
        out["is_signal"][i] = int(data["label_Hbb"][i])

        # Beregn jet_pt som pt for summen av partikkel-4-momenta
        out["jet_pt"][i] = pt(np.sum(Pmu_event, axis=0))

        # Legg til beam-partikler om ønskelig
        if add_beams:
            out["Pmu"][i, -nbeam:, :] = beam_vecs

        # Sett partikkel-etiketter: 1 for gyldige partikler, -1 for beams
        out["label"][i, :nobj] = 1
        if add_beams:
            out["label"][i, -nbeam:] = -1

        # Beregn invariant masse for alle partikler
        out["mass"][i, :] = masses(out["Pmu"][i, :, :])

        # Beregn evt. dot-produktmatrise
        if dot_products:
            out["dots"][i, :, :] = dots_matrix(out["Pmu"][i, :, :], out["Pmu"][i, :, :])

        # Hent de 10 jet-klassene og lagre i jet_label
        out["jet_label"][i, :] = np.array([
            data["label_QCD"][i],
            data["label_Hbb"][i],
            data["label_Hcc"][i],
            data["label_Hgg"][i],
            data["label_H4q"][i],
            data["label_Hqql"][i],
            data["label_Zqq"][i],
            data["label_Wqq"][i],
            data["label_Tbqq"][i],
            data["label_Tbl"][i]
        ], dtype=np.int16)

    # Lagre til HDF5-fil
    with h5py.File(output_file, "w") as f:
        for key in out:
            f.create_dataset(key, data=out[key], compression="gzip")
    print("Konvertert:", root_file, "->", output_file)


##########################################
# Hovedprogram                           #
##########################################

def main():
    folder = "/Users/danielsaga/Downloads/JetClass_Pythia_train_100M_part0"
    files = glob.glob(os.path.join(folder, "*.root"))
    start = time.time()
    for root_file in files:
        base = os.path.basename(root_file)
        output_file = os.path.join(folder, base.replace(".root", "_c.h5"))
        convert_root_to_lorentznet(root_file, output_file, add_beams=False, dot_products=False, double_precision=True)
    print("Total tid:", time.time() - start)


if __name__ == '__main__':
    main()
