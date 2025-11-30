from typing import Optional, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class Sequential(DllmAlgorithm):
    """Sequential unmasking algorithm.

    Unmasks tokens from left to right in a deterministic order.
    This strategy doesn't rely on confidence scores and is fully predictable.
    """

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        mask_index = forward_batch.input_ids == self.mask_id
        start = len(forward_batch.input_ids) - torch.sum(mask_index).item()

        # Track decoding order
        decoding_order = []

        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            # Forward pass to get predictions
            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            # Get predicted tokens
            x = torch.argmax(logits_output.full_logits, dim=-1)

            # Sequential selection: find first masked position
            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)

            # Find the leftmost (first) masked token
            masked_positions = torch.where(mask_index)[0]
            if len(masked_positions) > 0:
                # Select the first masked position
                first_masked_idx = masked_positions[0]
                transfer_index[first_masked_idx] = True

                # Track this position in decoding order
                decoding_order.append(first_masked_idx.item())

            # Unmask the selected token
            forward_batch.input_ids[transfer_index] = x[transfer_index]

        # Store decoding order in forward_batch
        forward_batch.dllm_decoding_order = decoding_order

        # Final forward pass with all tokens unmasked
        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        next_token_ids = forward_batch.input_ids[start:]
        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = Sequential
