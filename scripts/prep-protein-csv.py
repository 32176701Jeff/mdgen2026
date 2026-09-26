#!/usr/bin/env python3
"""Build a protein manifest from one PDB file per protein directory.

Expected input layout:

    INPUT_DIR/<protein_name>/<protein_name>.pdb

Output columns:

    name,seqres,position_ids
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


AMINO_ACID_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a name,seqres,position_ids CSV from "
            "INPUT_DIR/<protein_name>/<protein_name>.pdb files."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per protein.",
    )
    parser.add_argument(
        "--output",
        "--output_csv",
        dest="output_csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args()


def read_protein_residues(
    pdb_path: Path,
) -> list[tuple[str, int, str, str]]:
    """Return ordered unique (chain, resseq, insertion, resname) records."""
    residues: list[tuple[str, int, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    with pdb_path.open() as pdb_file:
        for line_number, line in enumerate(pdb_file, start=1):
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            resname = line[17:20].strip().upper()
            if resname not in AMINO_ACID_3TO1:
                continue

            chain = line[21].strip()
            resseq_text = line[22:26].strip()
            insertion = line[26].strip()
            residue_key = (chain, resseq_text, insertion)
            if residue_key in seen:
                continue
            seen.add(residue_key)

            if insertion:
                raise ValueError(
                    f"Insertion code {insertion!r} is not supported at "
                    f"{pdb_path}:{line_number} (residue {resseq_text}{insertion})."
                )
            try:
                resseq = int(resseq_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid PDB residue number {resseq_text!r} at "
                    f"{pdb_path}:{line_number}."
                ) from error

            residues.append((chain, resseq, insertion, resname))

    if not residues:
        raise ValueError(f"No supported protein residues found in {pdb_path}")
    return residues


def build_manifest_row(protein_name: str, pdb_path: Path) -> dict[str, str]:
    residues = read_protein_residues(pdb_path)
    protein_chains = {record[0] for record in residues}
    if len(protein_chains) != 1:
        display_chains = [chain if chain else "<blank>" for chain in protein_chains]
        raise ValueError(
            f"Expected one protein chain in {pdb_path}, found: "
            f"{', '.join(sorted(display_chains))}"
        )

    residue_numbers = [record[1] for record in residues]
    for previous, current in zip(residue_numbers, residue_numbers[1:]):
        if current <= previous:
            raise ValueError(
                f"Protein residue numbers must be strictly increasing in {pdb_path}; "
                f"found {previous} followed by {current}."
            )

    # position-id-csv
    minimum_position = min(residue_numbers)
    position_ids = [position - minimum_position for position in residue_numbers]
    seqres = "".join(AMINO_ACID_3TO1[record[3]] for record in residues)

    if len(seqres) != len(position_ids):
        raise ValueError(
            f"Sequence/position length mismatch in {pdb_path}: "
            f"seqres={len(seqres)}, position_ids={len(position_ids)}"
        )

    return {
        "name": protein_name,
        "seqres": seqres,
        "position_ids": json.dumps(position_ids, separators=(",", ":")),
    }


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    protein_dirs = sorted(path for path in args.input_dir.iterdir() if path.is_dir())
    if not protein_dirs:
        raise SystemExit(f"No protein subdirectories found in {args.input_dir}")

    rows: list[dict[str, str]] = []
    for protein_dir in protein_dirs:
        pdb_path = protein_dir / f"{protein_dir.name}.pdb"
        if not pdb_path.is_file():
            raise FileNotFoundError(f"Missing input PDB: {pdb_path}")

        row = build_manifest_row(protein_dir.name, pdb_path)
        rows.append(row)
        position_ids = json.loads(row["position_ids"])
        gap_count = sum(
            current - previous > 1
            for previous, current in zip(position_ids, position_ids[1:])
        )
        print(
            f"{protein_dir.name}: residues={len(row['seqres'])}, "
            f"position_range={position_ids[0]}..{position_ids[-1]}, "
            f"gaps={gap_count}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["name", "seqres", "position_ids"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {args.output_csv} ({len(rows)} proteins)")


if __name__ == "__main__":
    main()
