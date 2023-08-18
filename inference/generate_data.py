from datasets import load_dataset
from transformers import pipeline
import pyarrow
import argparse
import torch
import time

DATA_PATH = "datasets/"


def generate_data(dataset, pipe, num_outputs):
    for i in range(num_outputs):
        dataset[i]["text"] = "This is what you will do: \n" + \
            dataset[i]["text"]
        dataset[i]["output"] = pipe(dataset[i]["input"])[0]["generated_text"]
    return dataset


def save_data(dataset, output_file):
    table = pyarrow.Table.from_pydict(dataset)
    pyarrow.parquet.write_table(table, DATA_PATH + output_file)


def main(args):
    if args.dataset == "wikipedia":
        dataset = load_dataset(args.dataset, "20220301.en", streaming=True)
        dataset = dataset.remove_columns(["id", "url", "title"])
    else:
        dataset = load_dataset(args.dataset, streaming=True)
    dataset = dataset["train"]

    dataset = dataset.skip(args.skip_num_examples)

    pipe = pipeline("text-generation", model=args.model,
                    torch_dtype=torch.bfloat16)

    dataset = generate_data(dataset, pipe, args.num_generated_outputs)

    output_file = args.num_generated_outputs + time.strftime("%H%M%S")
    save_data(dataset, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_generated_outputs", type=int, default=1024)
    parser.add_argument("--skip_num_examples", type=int,
                        help="Number of examples to skip if you have already generated examples with the first n data already", default=0)

    args = parser.parse_args()
    main(args)
