import torch
from torch import nn
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(self.device).long()  # Ensure labels are on correct device and of type long
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits = logits.float()  # Ensure logits are of type float

        # Compute custom loss
        class_weights_tensor = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

    def set_class_weights(self, class_weights):
        self.class_weights = class_weights

    def set_device(self, device):
        self.device = device
