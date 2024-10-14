import torch
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        # Forward Pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits = logits.float()  # Ensure logits are in float32

        # Compute loss with class weights moved to the appropriate device
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float32).to(device=self.device))
        labels = labels.to(self.device).long()  # Ensure labels are in long (int64) type

        # Compute loss
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def set_device(self, device):
        self.device = device
