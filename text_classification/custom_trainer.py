import torch
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # Forward Pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Ensure logits are in float32 format
        logits = logits.float()

        # Move class weights to the correct device
        class_weights = torch.tensor(self.class_weights).to(self.device)

        # Compute Custom Loss
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def set_device(self, device):
        self.device = device
