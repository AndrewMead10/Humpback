import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from huggingface_hub import PyTorchModelHubMixin
from bitsandbytes.optim import GlobalOptimManager, Adam

# command for deepespeed
# sudo apt install nvidia-cuda-toolkit
# ps -aux|grep python to find instances to kill


class FinetuneBaseModel(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(self, model_name, lr, num_training_steps):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16)

        # convert for training with adam8bit
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32})

        self.lr = lr
        self.num_training_steps = num_training_steps

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), lr=self.lr, weight_decay=0.1, optim_bits=8)

        gamma = (self.lr-9e-6)/self.num_training_steps
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=gamma
        )
        return [optimizer], [scheduler]


class OASSTDatasetModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size, tune_on_output):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tune_on_output = tune_on_output

    def setup(self, stage=None):
        dataset = load_dataset("OpenAssistant/oasst1", split="train")

        dataset = dataset.filter(lambda example: example["lang"] == "en")
        dataset = self.filter_oasst_dataset(dataset)

        # use partial to pass tokenizer to tokenize_and_mask
        partial_tokenize_and_mask = partial(
            self.tokenize_and_mask, tokenizer=self.tokenizer, tune_on_output=self.tune_on_output)

        dataset = dataset.map(partial_tokenize_and_mask, batched=True)
        dataset = dataset.remove_columns(["instruction", "output"])

        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=20, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=20, pin_memory=True, collate_fn=self.collate_fn)

    def filter_oasst_dataset(self, dataset):
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

    def tokenize_and_mask(self, examples, tokenizer, tune_on_output):
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

    def collate_fn(self, batch):
        # Padding the input_ids and attention_masks using the tokenizer
        # batch = self.tokenizer.pad(batch, padding='longest', return_tensors="pt")
        batch[0]['input_ids'] = torch.tensor(
            batch[0]['input_ids']).unsqueeze(0)
        batch[0]['attention_mask'] = torch.tensor(
            batch[0]['attention_mask']).unsqueeze(0)
        batch[0]['labels'] = torch.tensor(batch[0]['labels']).unsqueeze(0)

        return batch[0]


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    num_gpus = 4
    batch_size = 1
    tune_on_output = False
    lr = 1e-5
    num_training_steps = 1000
    grad_acc = 32/(batch_size * num_gpus)

    dataset_module = OASSTDatasetModule(
        tokenizer=tokenizer, batch_size=batch_size, tune_on_output=tune_on_output)

    model = FinetuneBaseModel(
        model_name=model_name, lr=lr, num_training_steps=num_training_steps)

    # model = torch.compile(model)

    logger = WandbLogger(
        project="humpback base finetune", name="finetune_base_model_lightning")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=num_gpus,
        strategy="deepspeed_stage_3",
        precision="bf16-mixed",
        logger=logger,
        max_steps=num_training_steps,
        callbacks=[lr_monitor],
        accumulate_grad_batches=grad_acc
    )

    trainer.fit(model, dataset_module)

    model.save_to_hub("llama-7b-OASST-first-turn-high-quality")


if __name__ == "__main__":
    main()
