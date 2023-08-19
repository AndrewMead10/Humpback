from datasets import load_dataset
from transformers import pipeline
import argparse
import torch
import time


def generate_data(dataset, pipe, num_outputs, output_dir):
    for i in range(num_outputs):
        dataset[i]["text"] = "This is what you will do: \n" + \
            dataset[i]["text"]
        dataset[i]["output"] = pipe(dataset[i]["input"])[0]["generated_text"]

    dataset.save_to_disk(output_dir)
    return dataset


def main(args):
    if args.dataset == "wikipedia":
        dataset = load_dataset(args.dataset, "20220301.en", streaming=True)
        dataset = dataset.remove_columns(["id", "url", "title"])
    else:
        dataset = load_dataset(args.dataset, streaming=True)
    dataset = dataset["train"]

    dataset = dataset.skip(args.skip_num_examples)

    pipe = pipeline("text-generation", model=args.model,
                    torch_dtype=torch.bfloat16, device_map='auto')

    output_dir = f"{args.output_dir}/{args.dataset}_{args.skip_num_examples}_{args.skip_num_examples + args.num_outputs}"

    dataset = generate_data(dataset, pipe, args.num_outputs, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num_outputs", type=int,
                        default=1024, help="Number of examples to generate")
    parser.add_argument("--skip_num_examples", type=int,
                        help="Number of examples to skip if you have already generated examples with the first n data already", default=0)
    parser.add_argument("--output_dir", type=str,
                        help="Directory to save the generated data to", default="datasets/generated_data")

    args = parser.parse_args()
    main(args)
