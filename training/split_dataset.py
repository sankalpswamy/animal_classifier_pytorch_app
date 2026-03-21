from __future__ import annotations

import argparse

from data_utils import load_config, prepare_selected_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the selected 15 animal classes into train/val/test.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    prepare_selected_dataset(config)
    print("Dataset split complete.")


if __name__ == "__main__":
    main()
