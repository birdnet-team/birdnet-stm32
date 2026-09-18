"""CLI entry point for birdnet-stm32.

Usage:
    python -m birdnet_stm32 train ...
    python -m birdnet_stm32 convert ...
    python -m birdnet_stm32 equalize ...
    python -m birdnet_stm32 evaluate ...
    python -m birdnet_stm32 measure-operational ...
    python -m birdnet_stm32 deploy ...
    python -m birdnet_stm32 board-test ...
"""

import os
import sys


def main():
    """Dispatch to the appropriate CLI subcommand."""
    if len(sys.argv) < 2:
        print("Usage: birdnet-stm32 {train,convert,equalize,evaluate,measure-operational,deploy,board-test}")
        sys.exit(1)

    command = sys.argv[1]
    # Remove the subcommand from argv so argparse in each module sees correct args
    sys.argv = [f"birdnet-stm32 {command}"] + sys.argv[2:]

    if command == "train":
        # Must precede the first numpy import. Training forks loader workers from
        # a process that later runs numpy BLAS itself (the mel projection in
        # file-level validation); with a multithreaded OpenBLAS pool that main
        # process can deadlock after the fork. TensorFlow does not use
        # OpenBLAS, so one BLAS thread costs training nothing.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from birdnet_stm32.cli.train import main as train_main

        train_main()
    elif command == "convert":
        from birdnet_stm32.cli.convert import main as convert_main

        convert_main()
    elif command == "equalize":
        from birdnet_stm32.cli.equalize import main as equalize_main

        equalize_main()
    elif command == "evaluate":
        from birdnet_stm32.cli.evaluate import main as evaluate_main

        evaluate_main()
    elif command == "measure-operational":
        from birdnet_stm32.cli.measure_operational import main as measure_operational_main

        measure_operational_main()
    elif command == "deploy":
        from birdnet_stm32.cli.deploy import main as deploy_main

        deploy_main()
    elif command == "board-test":
        from birdnet_stm32.cli.board_test import main as board_test_main

        board_test_main()
    else:
        print(f"Unknown command: {command}")
        print("Usage: birdnet-stm32 {train,convert,equalize,evaluate,measure-operational,deploy,board-test}")
        sys.exit(1)


if __name__ == "__main__":
    main()
