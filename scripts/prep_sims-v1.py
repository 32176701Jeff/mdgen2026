import argparse
import os
from multiprocessing import Pool

import mdtraj
import numpy as np
import pandas as pd
import tqdm

from mdgen import residue_constants as rc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert trajectories to atom14 arrays without realignment."
    )
    parser.add_argument("--split", type=str, default="splits/atlas.csv")
    parser.add_argument(
        "--atlas_dir", type=str, default="/data/cb/scratch/datasets/atlas"
    )
    parser.add_argument("--outdir", type=str, default="./data_atlas")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--atlas", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


args = parse_args()

if args.stride <= 0:
    raise ValueError("--stride must be greater than zero")

os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col="name")
names = df.index


def traj_to_atom14(traj):
    arr = np.zeros(
        (traj.n_frames, traj.n_residues, 14, 3), dtype=np.float32
    )
    for residue_index, residue in enumerate(traj.top.residues):
        atom14_names = rc.restype_name_to_atom14_names.get(residue.name)
        if atom14_names is None:
            raise ValueError(
                f"Unsupported residue {residue.name!r} at topology index "
                f"{residue_index}"
            )

        for atom in residue.atoms:
            if atom.name not in atom14_names:
                print(f"Warning: {residue} atom {atom.name} not in atom14; skipped")
                continue
            atom14_index = atom14_names.index(atom.name)
            arr[:, residue_index, atom14_index] = (
                traj.xyz[:, atom.index] * 10.0
            )
    return arr


def validate_trajectory(traj, source):
    xyz = traj.xyz[0]
    unique_coordinates = np.unique(xyz, axis=0).shape[0]
    coordinate_std = xyz.std(axis=0)

    if traj.n_atoms > 1:
        first_distance = float(np.linalg.norm(xyz[0] - xyz[1]))
    else:
        first_distance = float("nan")

    print(
        f"{source}: frames={traj.n_frames}, atoms={traj.n_atoms}, "
        f"residues={traj.n_residues}, unique_xyz={unique_coordinates}, "
        f"xyz_std_nm={coordinate_std.tolist()}, "
        f"first_atom_distance_nm={first_distance:.6f}"
    )

    if unique_coordinates <= 1 or np.all(coordinate_std < 1e-6):
        raise ValueError(f"Collapsed coordinates detected in {source}")


def validate_atom14(arr, source):
    n_index = rc.atom_order["N"]
    ca_index = rc.atom_order["CA"]
    c_index = rc.atom_order["C"]

    n_ca = np.linalg.norm(
        arr[0, :, n_index] - arr[0, :, ca_index], axis=-1
    )
    ca_c = np.linalg.norm(
        arr[0, :, ca_index] - arr[0, :, c_index], axis=-1
    )
    valid = (n_ca > 0) & (ca_c > 0)

    if not np.any(valid):
        raise ValueError(f"No valid backbone bond distances found in {source}")

    print(
        f"{source}: atom14_shape={arr.shape}, "
        f"valid_backbones={int(valid.sum())}/{arr.shape[1]}, "
        f"mean_N_CA_A={float(n_ca[valid].mean()):.4f}, "
        f"mean_CA_C_A={float(ca_c[valid].mean()):.4f}"
    )

    if float(n_ca[valid].mean()) < 0.5 or float(ca_c[valid].mean()) < 0.5:
        raise ValueError(f"Collapsed atom14 backbone detected in {source}")


def convert_trajectory(xtc_path, pdb_path, out_path):
    # Apply stride while reading so a large trajectory is not fully loaded.
    traj = mdtraj.load(
        xtc_path,
        top=pdb_path,
        stride=args.stride,
    )
    traj.atom_slice(
        [atom.index for atom in traj.top.atoms if atom.element.symbol != "H"],
        inplace=True,
    )

    # These trajectories contain repeated PDB frames and are already aligned.
    # Calling traj.superpose(traj) is unnecessary and can be costly for Piezo.
    validate_trajectory(traj, xtc_path)
    arr = traj_to_atom14(traj)
    validate_atom14(arr, xtc_path)
    np.save(out_path, arr)
    print(f"Saved {out_path}")


def do_job(name):
    pdb_path = f"{args.atlas_dir}/{name}/{name}.pdb"

    if args.atlas:
        for replicate in (1, 2, 3):
            xtc_path = (
                f"{args.atlas_dir}/{name}/"
                f"{name}_prod_R{replicate}_fit.xtc"
            )
            out_path = (
                f"{args.outdir}/{name}_R{replicate}{args.suffix}.npy"
            )
            convert_trajectory(xtc_path, pdb_path, out_path)
    else:
        xtc_path = f"{args.atlas_dir}/{name}/{name}.xtc"
        out_path = f"{args.outdir}/{name}{args.suffix}.npy"
        convert_trajectory(xtc_path, pdb_path, out_path)


def expected_outputs(name):
    if args.atlas:
        return [
            f"{args.outdir}/{name}_R{i}{args.suffix}.npy" for i in (1, 2, 3)
        ]
    return [f"{args.outdir}/{name}{args.suffix}.npy"]


def main():
    jobs = [
        name
        for name in names
        if not all(os.path.exists(path) for path in expected_outputs(name))
    ]

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            for _ in tqdm.tqdm(
                pool.imap(do_job, jobs), total=len(jobs)
            ):
                pass
    else:
        for _ in tqdm.tqdm(map(do_job, jobs), total=len(jobs)):
            pass


if __name__ == "__main__":
    main()
