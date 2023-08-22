from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm
from functools import partial
import argparse

from accelerate import Accelerator


def filter_oasst_dataset(dataset):
    new_dataset = {"instruction": [], "output": []}
    print("getting first turns")
    for i in tqdm(range(len(dataset))):
        if dataset[i]["parent_id"] == None:
            child = dataset[i+1]
            assert child["parent_id"] == dataset[i]["message_id"]
            if child["rank"] == 0:
                new_dataset["instruction"].append(dataset[i]["text"])
                new_dataset["output"].append(child["text"])

    # convert dict to hf dataset
    new_dataset = Dataset.from_dict(new_dataset)
    # should be ~3200
    print(len(new_dataset))
    return new_dataset


def tokenize_and_mask(examples, tokenizer):
    """
    Tokenizes combined prompt and response and generates attention masks so that we only attend to the response.

    Args:
    - examples (dict): Dictionary with keys 'instruction' and 'output' containing lists of strings.

    Returns:
    - List of dictionaries containing tokenized 'input_ids' and 'attention_mask' for each combined string.
    """

    combined_texts = [prompt + ' ' + response for prompt,
                      response in zip(examples['instruction'], examples['output'])]

    # Tokenize the combined texts and responses
    encoding = tokenizer(combined_texts, truncation=True,
                         padding=True, return_tensors="pt")
    response_encodings = tokenizer(
        examples['output'], truncation=True, padding=True, return_tensors="pt", add_special_tokens=False)

    response_lengths = torch.tensor(
        [len(enc) for enc in response_encodings['input_ids']])

    # Calculate mask_start_positions (adjust for [CLS] and [SEP] tokens by adding 2 to the offset)
    mask_start_positions = encoding['attention_mask'].sum(
        dim=1) - response_lengths + 1

    # Use the original attention mask and update it
    idx_matrix = torch.arange(encoding['attention_mask'].size(
        1)).unsqueeze(0).to(mask_start_positions.device)
    mask_update = (idx_matrix < mask_start_positions.unsqueeze(1)).long()

    # Only update the mask where necessary
    encoding['attention_mask'] *= mask_update

    # Convert tensors to lists
    encoding['input_ids'] = encoding['input_ids'].tolist()
    encoding['attention_mask'] = encoding['attention_mask'].tolist()

    return encoding


def main(args):

    accelerator = Accelerator()
    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16)
    model = accelerator.prepare(model)

    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    dataset = dataset.filter(lambda example: example["lang"] == "en")
    dataset = filter_oasst_dataset(dataset)

    # use partial to pass tokenizer to tokenize_and_mask
    partial_tokenize_and_mask = partial(tokenize_and_mask, tokenizer=tokenizer)

    dataset = dataset.map(partial_tokenize_and_mask, batched=True)
    dataset = dataset.remove_columns(["instruction", "output"])

    num_devices = torch.cuda.device_count()
    bs = 32
    per_device_bs = bs // num_devices

    # lr= 1e − 5 which linearly decays to 9e − 6 at the end of training
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        overwrite_output_dir=True,
        max_steps=500,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=per_device_bs,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_steps=100,
        fp16=True,
        tf32=True,
        torch_compile=True,
        save_steps=250,
        save_total_limit=1,
        report_to="wandb",
        run_name="oasst finetune",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-2-7b-hf")
    args = parser.parse_args()
    main(args)
