"""CLI entry point for birdnet-stm32.

Usage:
    python -m birdnet_stm32 train ...
    python -m birdnet_stm32 convert ...
    python -m birdnet_stm32 evaluate ...
    python -m birdnet_stm32 deploy ...
"""

import sys


def main():
    """Dispatch to the appropriate CLI subcommand."""
    if len(sys.argv) < 2:
        print("Usage: birdnet-stm32 {train,convert,evaluate,deploy}")
        sys.exit(1)

    command = sys.argv[1]
    # Remove the subcommand from argv so argparse in each module sees correct args
    sys.argv = [f"birdnet-stm32 {command}"] + sys.argv[2:]

    if command == "train":
        from birdnet_stm32.cli.train import main as train_main

        train_main()
    elif command == "convert":
        from birdnet_stm32.cli.convert import main as convert_main

        convert_main()
    elif command == "evaluate":
        from birdnet_stm32.cli.evaluate import main as evaluate_main

        evaluate_main()
    elif command == "deploy":
        from birdnet_stm32.cli.deploy import main as deploy_main

        deploy_main()
    else:
        print(f"Unknown command: {command}")
        print("Usage: birdnet-stm32 {train,convert,evaluate,deploy}")
        sys.exit(1)


if __name__ == "__main__":
    main()
