import argparse
from timeit import default_timer as timer
from tensorizer import tensorize, foo

def run(args: argparse.Namespace):
    start_time = timer()
    try:
        foo()
    except Exception as e:
        print(f"Exception occured: {e}")

    end_time = timer()
    elapsed_time = end_time - start_time
    print(f"Finished operation in {elapsed_time:.2f} sec")

if __name__ == "__main__":
    args = parse_args()
    run(args)
