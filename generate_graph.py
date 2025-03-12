import argparse
import yaml
from utils.graph_constructor import GraphConstructor


def main():
    parser = argparse.ArgumentParser(description="Graph Data Construction")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Dataset raw path")
    parser.add_argument("--processed_dir", type=str, required=True,
                        help="Processed graph data path")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config file path")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    constructor = GraphConstructor(config, args.dataset_dir, args.processed_dir)
    constructor.process_dataset()


if __name__ == "__main__":
    main()