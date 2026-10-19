"""Import the Candombe beat/downbeat dataset (Universidad de la República) as training pieces.

Dataset: https://zenodo.org/records/6533068 (35 performances, ~2 h, expert
beat/downbeat annotations; Nunes, Rocamora, Jure & Biscainho, ISMIR 2015).
License CC BY-NC-SA 4.0: non-commercial use, attribution, share-alike.

The recordings are mixes of 3-5 drums, so instruments are still separated by
band filters (heuristic), but the bar grid comes from the expert annotations.

    python src/import_candombe_dataset.py --download --dataset-dir data/raw/candombe
    python src/import_candombe_dataset.py --dataset-dir data/raw/candombe
"""

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from analyze_reference import analyze_reference, save_outputs

ZENODO_RECORD = "https://zenodo.org/api/records/6533068"
AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".ogg")


def download(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_RECORD) as response:
        record = json.load(response)
    for entry in record.get("files", []):
        name = entry["key"]
        target = dataset_dir / name
        if not target.exists():
            print(f"Downloading {name} ({entry.get('size', 0) / 1e6:.0f} MB)")
            urllib.request.urlretrieve(entry["links"]["self"], target)
        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as archive:
                for member in archive.namelist():
                    # Never write outside dataset_dir.
                    if member.startswith("/") or ".." in Path(member).parts:
                        raise ValueError(f"Unsafe path in archive: {member}")
                archive.extractall(dataset_dir)


def pair_files(dataset_dir: Path) -> List[Tuple[Path, Optional[Path]]]:
    """Match every audio file with the annotation CSV that shares its stem."""
    annotations = {path.stem: path for path in dataset_dir.rglob("*.csv")}
    pairs = []
    for audio in sorted(p for p in dataset_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES):
        match = annotations.get(audio.stem) or next((path for stem, path in annotations.items() if stem.startswith(audio.stem)), None)
        pairs.append((audio, match))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw/candombe"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/candombe_dataset"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("data/analysis/candombe_dataset"))
    parser.add_argument("--download", action="store_true", help="Download the dataset from Zenodo first (CC BY-NC-SA 4.0).")
    args = parser.parse_args()

    if args.download:
        download(args.dataset_dir)

    pairs = pair_files(args.dataset_dir)
    if not pairs:
        raise FileNotFoundError(f"No audio found under {args.dataset_dir}")

    imported = 0
    for audio, beats in pairs:
        if beats is None:
            print(f"SKIP {audio.name}: no annotation CSV")
            continue
        summary, events, arrays = analyze_reference(audio_path=str(audio), beats_file=str(beats))
        save_outputs(summary, events, arrays, str(args.analysis_dir / f"{audio.stem}.json"), str(args.out_dir / f"{audio.stem}.npz"))
        print(f"{audio.name}: {summary['bars']} compases, {summary['tempo_start']} -> {summary['tempo_end']} BPM, {summary['counts_by_instrument']}")
        imported += 1
    print(f"Imported {imported}/{len(pairs)} recordings into {args.out_dir}")


if __name__ == "__main__":
    main()
