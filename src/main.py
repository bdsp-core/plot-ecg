import argparse
from timeit import default_timer as timer
from arguments import parse_args
from tensorizer import tensorize


def run(args: argparse.Namespace):
    start_time = timer()
    tensorize(
        xml=args.xml,
        hd5=args.hd5,
    )
    end_time = timer()
    elapsed_time = end_time - start_time
    print(f"Finished operation in {elapsed_time:.2f} sec")


if __name__ == "__main__":
    args = parse_args()
    run(args)
