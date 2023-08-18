from datasets import load_dataset
from transformers import pipeline

rating_prompt = """Below is an instruction from an user and a candidate answer. Evaluate whether or not the answer is a good example of how AI Assistant should respond to the user's instruction. Please assign a score using the following 5-point scale:
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

DATA_PATH = "datasets/"
pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")

dataset = load_dataset("data.paraquet")

for i in range(len(dataset)):
    evaluation_string = rating_prompt.format(
        generated_instruction=dataset[i]["text"],
        output=pipe(dataset[i]["input"])[0]["generated_text"]
    )

    dataset[i]["rating"] = pipe(evaluation_string)[0]["generated_text"]

dataset.save_to_disk(DATA_PATH + "data_with_ratings.parquet")
