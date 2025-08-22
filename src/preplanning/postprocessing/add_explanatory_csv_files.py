# use this script like this: python -m add_explanatory_csv_files Path/to/Folders
# it will go through all files in all subfolders and will create a short per folder overview of fileID, control strategy, objective / pareto constraint.

#!/usr/bin/env python3
import os
import h5py
import csv
import argparse
import logging
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan subfolders for HDF5 files, read 'Comment', and write per-folder summary.csv."
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing subfolders with .h5/.hdf5 files",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s: %(message)s",
    )


def read_comment_dataset(h5_path: Path) -> str:
    """Read the 'Comment' dataset as a string. Raises KeyError if not found."""
    with h5py.File(h5_path, "r") as f:
        if "Comment" not in f:
            raise KeyError("'Comment' dataset not found")
        data = f["Comment"][()]
        return (
            data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
        )


def parse_comment(comment: str):
    """
    Parse control strategy and objective/constraint from a comment string
    like: 'distributed CPC, energy costs ub: 6000.0'
    """
    parts = [p.strip() for p in comment.split(",")]
    control_strategy = parts[0] if len(parts) > 0 else ""
    objective = parts[1] if len(parts) > 1 else ""

    # replace ub: with <=, lb: with >=
    objective = objective.replace("ub:", "<=").replace("lb:", ">=")
    return control_strategy, objective


def process_folder(root_dir: Path):
    for subdir, _, files in os.walk(root_dir):
        subdir_path = Path(subdir)
        hdf5_files = [f for f in files if f.lower().endswith((".h5", ".hdf5"))]
        if not hdf5_files:
            logging.debug("No HDF5 files in %s; skipping", subdir_path)
            continue

        csv_path = subdir_path / "summary.csv"
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ["file ID", "control strategy", "objective / additional constraint"]
                )

                for file in sorted(hdf5_files):
                    file_id = os.path.splitext(file)[0]
                    file_path = subdir_path / file

                    try:
                        comment = read_comment_dataset(file_path)
                        control_strategy, objective = parse_comment(comment)
                        writer.writerow([file_id, control_strategy, objective])
                        logging.debug("Wrote summary for %s", file_path)
                    except KeyError as e:
                        logging.warning("Skipping %s: %s", file_path, e)
                    except Exception as e:
                        logging.error("Error reading %s: %s", file_path, e)
            logging.info("Created %s", csv_path)
        except Exception as e:
            logging.error("Failed to write %s: %s", csv_path, e)


def main():
    args = parse_args()
    setup_logging(args.log_level)

    root = Path(args.root_dir).expanduser().resolve()
    if not root.exists():
        logging.error("Path does not exist: %s", root)
        return
    if not root.is_dir():
        logging.error("Path is not a directory: %s", root)
        return

    logging.info("Scanning root: %s", root)
    process_folder(root)


if __name__ == "__main__":
    main()
