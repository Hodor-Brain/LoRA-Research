# Inference Engine Logic

import logging
from typing import Optional, List, Dict, Any
import torch
from collections import defaultdict

from managers.adapter_manager import LoRAAdapterManager
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 adapter_manager: LoRAAdapterManager,
                 device: str = 'cuda'):
        self.base_model = model
        self.tokenizer = tokenizer
        self.adapter_manager = adapter_manager
        self.device = device
        if self.tokenizer.padding_side != 'left':
             logger.warning(f"Tokenizer padding side was not 'left', setting it to 'left' for batch generation.")
             self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
                 logger.warning("Tokenizer missing pad token in InferenceEngine, setting to EOS.")
            else:
                 logger.error("Tokenizer missing both pad and EOS token in InferenceEngine. Batch inference might fail.")
                 raise ValueError("Tokenizer missing both pad and EOS token in InferenceEngine.")

    def process_batch(self, inference_requests: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Processes a batch of inference requests, handling multiple adapters.

        Groups requests by adapter path, loads the required adapter (using cache),
        runs generation for each group, and returns results in the original order.

        Args:
            inference_requests (List[Dict[str, Any]]): A list of dictionaries,
                each containing at least 'prompt' and optionally 'adapter_path'
                and 'max_new_tokens'.

        Returns:
            List[Optional[str]]: A list of generated texts (excluding prompts),
                                matching the order of input requests. None for failed requests.
        """
        if not inference_requests:
            return []

        results = [None] * len(inference_requests)
        grouped_requests = defaultdict(list)
        original_indices = defaultdict(list)

        for i, req in enumerate(inference_requests):
            adapter_path = req.get('adapter_path')
            grouped_requests[adapter_path].append(req['prompt'])
            original_indices[adapter_path].append(i)

        logger.info(f"Processing batch of {len(inference_requests)} requests, grouped into {len(grouped_requests)} sub-batches by adapter.")

        for adapter_path, prompts in grouped_requests.items():
            logger.info(f"Processing sub-batch for adapter: '{adapter_path if adapter_path else 'Base Model'}' ({len(prompts)} prompts)")
            
            target_model = self.base_model
            if adapter_path:
                model_with_adapter = self.adapter_manager.load_inference_adapter(adapter_path)
                if model_with_adapter is None:
                    logger.error(f"Failed to load adapter {adapter_path}. Skipping sub-batch.")
                    continue
                target_model = model_with_adapter
            
            # TODO: A more robust approach might average or validate settings across the group
            first_req_idx = original_indices[adapter_path][0]
            max_new_tokens = inference_requests[first_req_idx].get('max_new_tokens', 50)

            try:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                input_ids_len = inputs['input_ids'].shape[1] 

                with torch.no_grad():
                    outputs = target_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                batch_output_ids = outputs[:, input_ids_len:]
                generated_texts = self.tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)

                indices_for_group = original_indices[adapter_path]
                for i, generated_text in enumerate(generated_texts):
                    original_index = indices_for_group[i]
                    results[original_index] = generated_text
                logger.info(f"Sub-batch for adapter '{adapter_path if adapter_path else 'Base Model'}' processed successfully.")

            except Exception as e:
                logger.error(f"Error processing sub-batch for adapter '{adapter_path if adapter_path else 'Base Model'}': {e}", exc_info=True)

        logger.info(f"Batch processing finished. Returning {len(results)} results.")
        return results

    def run_inference_single(self, prompt: str, adapter_path: Optional[str] = None, max_new_tokens: int = 50) -> Optional[str]:
        """Runs inference for a single prompt, potentially using a LoRA adapter.

        Args:
            prompt (str): The input text prompt.
            adapter_path (Optional[str]): Path to the LoRA adapter directory. 
                                          If None, uses the base model.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            Optional[str]: The generated text (excluding the prompt), or None on error.
        """
        logger.info(f"Running single inference for adapter: '{adapter_path if adapter_path else 'Base Model'}'")
        
        target_model = self.base_model
        if adapter_path:
            model_with_adapter = self.adapter_manager.load_inference_adapter(adapter_path)
            if model_with_adapter is None:
                logger.error(f"Failed to get model instance for adapter {adapter_path} from manager.")
                return None
            target_model = model_with_adapter
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_ids_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                outputs = target_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            output_ids = outputs[0][input_ids_len:]
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            logger.info(f"Inference successful. Generated {len(output_ids)} tokens.")
            return generated_text

        except Exception as e:
            logger.error(f"Error during single inference: {e}", exc_info=True)
            return None 