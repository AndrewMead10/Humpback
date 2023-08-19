from datasets import load_from_disk
from transformers import pipeline
import argparse
import torch

RATING_PROMPT = """Below is an instruction from an user and a candidate answer. Evaluate whether or not the answer is a good example of how AI Assistant should respond to the user's instruction. Please assign a score using the following 5-point scale:
1: It means the answer is incomplete, vague, off-topic, controversial, or not exactly what the user asked for. For example, some content seems missing, numbered list does not start from the beginning, the opening sentence repeats user's question. Or the response is from another person's perspective with their personal experience (e.g. taken from blog posts), or looks like an answer from a forum. Or it contains promotional text, navigation text, or other irrelevant information.
2: It means the answer addresses most of the asks from the user. It does not directly address the user's question. For example, it only provides a high-level methodology instead of the exact solution to user's question.
3: It means the answer is helpful but not written by an AI Assistant. It addresses all the basic asks from the user. It is complete and self contained with the drawback that the response is not written from an AI assistant's perspective, but from other people's perspective. The content looks like an excerpt from a blog post, web page, or web search results. For example, it contains personal experience or opinion, mentions comments section, or share on social media, etc.
4: It means the answer is written from an AI assistant's perspective with a clear focus of addressing the instruction. It provide a complete, clear, and comprehensive response to user's question or instruction without missing or irrelevant information. It is well organized, self-contained, and written in a helpful tone. It has minor room for improvement, e.g. more concise and focused.
5: It means it is a perfect answer from an AI Assistant. It has a clear focus on being a helpful AI Assistant, where the response looks like intentionally written to address the user's question or instruction without any irrelevant sentences. The answer provides high quality content, demonstrating expert knowledge in the area, is very well written, logical, easy-to-follow, engaging and insightful.
Please first provide a brief reasoning you used to derive the rating score, and
then write "Score: <rating>" in the last line.

{generated_instruction}
{output}
"""


def rate_dataset(dataset_path, pipe):
    dataset = load_from_disk(dataset_path)

    for i in range(len(dataset)):
        evaluation_string = RATING_PROMPT.format(
            generated_instruction=dataset[i]["text"],
            output=pipe(dataset[i]["input"])
        )

        # evaluate the prompt with the model and then split the text on the last word in dataset[i]["input"] to get just the rating text
        dataset[i]["rating"] = pipe(evaluation_string)[0]["generated_text"].split(
            dataset[i]["input"].split(" ")[-1])[1]

    dataset.save_to_disk(dataset_path)


def main(args):

    pipe = pipeline("text-generation", model=args.model,
                    torch_dtype=torch.bfloat16, device_map='auto')

    rate_dataset(args.dataset_path, pipe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        default="datasets/generated_data/wikipedia_0_1024")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-2-7b-hf")

    args = parser.parse_args()
    main(args)
