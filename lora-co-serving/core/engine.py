# Inference Engine Logic

import logging
from typing import Optional, List, Dict, Any, Tuple
import torch
from collections import defaultdict
import time

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

    def _group_requests_by_adapter(self, batch: List[Dict[str, Any]]) -> Dict[Optional[str], List[Dict[str, Any]]]:
        """Groups incoming requests by their specified adapter_path."""
        grouped = {}
        for request in batch:
            adapter_path = request.get('adapter_path') # None means use base model
            if adapter_path not in grouped:
                grouped[adapter_path] = []
            grouped[adapter_path].append(request)
        return grouped

    def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a batch of inference requests, handling adapter loading/unloading."""
        if not batch:
            return []

        results = [] 
        grouped_requests = self._group_requests_by_adapter(batch)
        logger.info(f"Processing batch of {len(batch)} requests, grouped into {len(grouped_requests)} sub-batches by adapter.")

        original_adapter_state = self.adapter_manager.get_active_adapters()
        
        try:
            for adapter_path, requests in grouped_requests.items():
                sub_batch_results = self._process_sub_batch(adapter_path, requests)
                results.extend(sub_batch_results)
        finally:
            # Restore original adapter state if needed (e.g., if training was interrupted)
            # This simple restoration might need refinement in complex scenarios
            # logger.debug(f"Attempting to restore original adapter state: {original_adapter_state}")
            # current_adapters = self.adapter_manager.get_active_adapters()
            # if current_adapters != original_adapter_state:
            #     logger.warning("Adapters changed during inference, attempting restore - this may be complex!")
                # Simple approach: unload all, reload originals? Needs careful thought.
                # self.adapter_manager.unload_all_adapters() # Risky if base model depends on adapter?
                # for adapter_id in original_adapter_state:
                #     self.adapter_manager.load_inference_adapter(adapter_id, ???) # Need path! 
            pass # Simplified for now

        logger.info(f"Batch processing finished. Returning {len(results)} results.")
        return results

    def _process_sub_batch(self, adapter_path: Optional[str], requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a sub-batch of requests that all use the same LoRA adapter (or base model)."""
        adapter_id = self.adapter_manager.get_adapter_id_from_path(adapter_path) if adapter_path else "Base Model"
        logger.info(f"Processing sub-batch for adapter: '{adapter_id}' ({len(requests)} prompts)")
        sub_batch_results = []
        
        # Prepare for generation
        prompts = [req['prompt'] for req in requests]
        generation_params = requests[0].get('generation_params', {})
        
        # Ensure the correct adapter is active OR that adapters are disabled for base model
        # This call handles deleting previous training adapter if necessary
        activated_model = self.adapter_manager.load_inference_adapter(adapter_path)
        success = activated_model is not None
        
        if not success:
            logger.error(f"Failed to set adapter '{adapter_id}' for inference. Skipping sub-batch.")
            # Return errors for all requests in this sub-batch
            processing_failed_time = time.time()
            for req in requests:
                sub_batch_results.append({
                    'request_id': req['request_id'],
                    'submission_time': req.get('submission_time'),
                    'processing_start_time': None,
                    'processing_end_time': processing_failed_time,
                    'output': None,
                    'error': f"Failed to load adapter '{adapter_id}'"
                })
            return sub_batch_results

        try:
            # Record start time just before tokenization/generation
            processing_start_time = time.time()
            
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # Generate
            # load_inference_adapter should have handled disabling any previous training adapter
            # No specific adapter is set if adapter_path is None
            with torch.no_grad():
                    outputs = self.base_model.generate(
                        **inputs,
                        max_new_tokens=generation_params.get('max_new_tokens', 50),
                        temperature=generation_params.get('temperature', 0.7),
                        do_sample=generation_params.get('do_sample', True),
                        pad_token_id=self.tokenizer.pad_token_id
                    )
            
            # Record end time immediately after generation
            processing_end_time = time.time()
            
            # Decode results
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Store results with timing information
            for i, req in enumerate(requests):
                # Simple way to get only the generated part (might need adjustment based on model/tokenizer)
                output_text = decoded_outputs[i][len(prompts[i]):] if len(decoded_outputs[i]) > len(prompts[i]) else decoded_outputs[i]
                
                sub_batch_results.append({
                    'request_id': req['request_id'],
                    'submission_time': req.get('submission_time'),
                    'processing_start_time': processing_start_time,
                    'processing_end_time': processing_end_time,
                    'output': output_text.strip(),
                    'error': None
                })
            logger.info(f"Sub-batch for adapter '{adapter_id}' processed successfully.")

        except Exception as e:
            logger.error(f"Error processing sub-batch for adapter '{adapter_id}': {e}", exc_info=True)
            processing_failed_time = time.time()
            # Return errors for all requests in this sub-batch
            sub_batch_results = [] # Clear any partial results
            for req in requests:
                 sub_batch_results.append({
                    'request_id': req['request_id'],
                    'submission_time': req.get('submission_time'),
                    'processing_start_time': None, # Indicate failure before processing started/finished
                    'processing_end_time': processing_failed_time,
                    'output': None,
                    'error': str(e)
                })
                
        return sub_batch_results

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