import argparse
import yaml
from utils.graph_constructor import GraphConstructor


def main():
    parser = argparse.ArgumentParser(description="Graph Data Construction")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config file path")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    constructor = GraphConstructor(config)
    constructor.process_dataset()


if __name__ == "__main__":
    main()