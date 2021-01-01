# Imports: standard library
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hd5",
        type=str,
        help="",
    )
    parser.add_argument(
        "--plot",
        type=str,
        help="",
    )
    parser.add_argument(
        "--xml",
        type=str,
        help="",
    )
    args = parser.parse_args()
    _process_args(args)
    return args


def _process_args(args: argparse.Namespace):
    if not os.path.exists(args.xml):
        raise ValueError(f"{args.xml_folder} does not exist")
    if not os.path.exists(args.hd5):
        raise ValueError(f"{args.hd5} does not exist")
    if not os.path.exists(args.plot):
        raise ValueError(f"{args.plot} does not exist")
