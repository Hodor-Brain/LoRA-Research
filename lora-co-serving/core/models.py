# Base Model Loading Logic

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def load_base_model(model_name_or_path: str, device: str = 'cuda'):
    """Loads the base LLM and tokenizer onto the specified device.

    Args:
        model_name_or_path (str): The Hugging Face model ID or local path.
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the loaded model and tokenizer, or (None, None) on error.
    """
    logger.info(f"Attempting to load base model: {model_name_or_path}...")
    try:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Using torch dtype: {torch_dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.warning(f"Tokenizer missing pad token, setting to EOS token: {tokenizer.pad_token}")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                logger.warning("Tokenizer missing both pad and EOS token. Added '[PAD]' as pad token and resized model embeddings.")

        model.to(device)
        logger.info(f"Successfully loaded model {model_name_or_path} to {device}.")
        logger.info(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model {model_name_or_path}: {e}", exc_info=True)
        return None, None 