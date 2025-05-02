import argparse
from config.setting import ROOT_DIR
from utils.merge import adapter_merge


def main(args):
    adapter_merge(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="test", description="Testing about Conversational Context Inference."
    )
    args = parser.add_argument_group("Common Parameter")
    args.add_argument("--model", type=str, required=True)
    args.add_argument("--adapter_path", type=str, required=True)
    args.add_argument("--merge_output_dir", type=str, required=True)
    args.add_argument("--max_seq_len", type=int, required=True)
    main(parser.parse_args())
