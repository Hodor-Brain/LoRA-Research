# Training Step Logic

import logging
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, PeftModel
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)

class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: PreTrainedTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        # Tokenize upfront? Or lazily?
        # For simplicity now, let's tokenize lazily in __getitem__
        # More efficient would be to pre-tokenize
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.warning("Pad token set to EOS token for SimpleTextDataset tokenizer")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].squeeze(0) 
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def setup_lora_training(base_model: PreTrainedModel, tokenizer: PreTrainedTokenizer, lora_config_dict: Dict[str, Any]) -> Optional[PeftModel]:
    """Applies LoRA configuration to the base model for training.

    Args:
        base_model (PreTrainedModel): The base language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        lora_config_dict (Dict[str, Any]): Dictionary with PEFT LoraConfig parameters 
                                           (e.g., r, lora_alpha, target_modules).

    Returns:
        Optional[PeftModel]: The PEFT model ready for training, or None on error.
    """
    try:
        lora_config_dict.setdefault("task_type", "CAUSAL_LM")
        peft_config = LoraConfig(**lora_config_dict)
        
        peft_model = get_peft_model(base_model, peft_config)

        peft_model = get_peft_model(base_model, peft_config)
        peft_model.train()
        peft_model.print_trainable_parameters()
        logger.info("PEFT model created and set to training mode.")
        return peft_model

    except Exception as e:
        logger.error(f"Error setting up LoRA training: {e}", exc_info=True)
        return None

def execute_training_step(
    model: PeftModel, 
    optimizer: torch.optim.Optimizer, 
    batch: Dict[str, torch.Tensor], 
    device: str = 'cuda'
) -> Optional[float]:
    """Executes a single training step (forward, backward, optimizer step).

    Args:
        model (PeftModel): The PEFT model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        batch (Dict[str, torch.Tensor]): A batch of data (containing input_ids, attention_mask, labels).
        device (str): The device to run the computation on.

    Returns:
        Optional[float]: The loss value for the step, or None on error.
    """
    model.train()
    optimizer.zero_grad()

    try:
        inputs = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**inputs)
        loss = outputs.loss

        if loss is None:
            logger.error("Model did not return loss. Check if labels are provided correctly.")
            return None

        loss.backward()
        optimizer.step()

        return loss.item()

    except Exception as e:
        logger.error(f"Error during training step: {e}", exc_info=True)
        return None
