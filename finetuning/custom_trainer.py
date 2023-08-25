from transformers import Trainer
from torch import nn
import torch


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        # linear lr decay from 1e-5 to 9e-6
        gamma = (1e-5-9e-6)/num_training_steps
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=gamma
        )
