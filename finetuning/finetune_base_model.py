from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm
from functools import partial
import argparse

from accelerate import Accelerator


def filter_oasst_dataset(dataset):
    new_dataset = {"instruction": [], "output": []}
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
    print(f"length of dataset: {len(new_dataset)}")
    return new_dataset


def tokenize_and_mask(examples, tokenizer, tune_on_output):
    # tune_on_output: if True, only attend to the output and ignore the instruction, if False, only attend to the instruction and ignore the output

    examples['instruction'] = [instruction +
                               " Answer in the style of an AI Assistant." for instruction in examples['instruction']]

    combined_texts = [prompt + ' ' + response for prompt,
                      response in zip(examples['instruction'], examples['output'])]

    # Tokenize the combined texts and responses
    encoding = tokenizer(combined_texts, truncation=True,
                         padding=True, return_tensors="pt")
    instruction_encodings = tokenizer(
        examples['instruction'], truncation=True, padding=True, return_tensors="pt", add_special_tokens=False)

    # add one to account for the BOS token at the start of the instruction
    instruction_lengths = torch.Tensor(
        [len(enc)+1 for enc in instruction_encodings['input_ids']])

    # only attent to the instruction and ignore the response
    mask = torch.arange(
        encoding['attention_mask'].size(1)).unsqueeze(0) < instruction_lengths.unsqueeze(-1)

    if tune_on_output:
        mask = ~mask

    # Create a tensor of -100s with the same shape as encoding['input_ids']
    negative_ones = torch.full_like(encoding['input_ids'], -100)

    # Use the mask to replace the values of encoding['input_ids'] where the mask is 0
    encoding['labels'] = torch.where(
        mask, encoding['input_ids'], negative_ones)

    encoding['input_ids'] = encoding['input_ids'].tolist()
    encoding['attention_mask'] = encoding['attention_mask'].tolist()
    encoding['labels'] = encoding['labels'].tolist()

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
    partial_tokenize_and_mask = partial(
        tokenize_and_mask, tokenizer=tokenizer, tune_on_output=args.tune_on_output)

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
        push_to_hub=True,
        hub_model_id="Llama-2-OASST-First-Turn-Instruction-Only",
        hub_strategy="every_save",
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
    parser.add_argument("--tune_on_output", action="store_true")
    args = parser.parse_args()
    main(args)
