from typing import Optional, Tuple, Union, List

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class SequentialVerbose(DllmAlgorithm):
    """Sequential unmasking algorithm with verbose output.

    Unmasks tokens from left to right in a deterministic order.
    Outputs the decoding order for debugging and analysis.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoding_order = []  # Track decoding order

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        mask_index = forward_batch.input_ids == self.mask_id
        start = len(forward_batch.input_ids) - torch.sum(mask_index).item()

        # Reset decoding order for this generation
        self.decoding_order = []

        print("\n" + "="*60)
        print("SEQUENTIAL DLLM DECODING - Verbose Mode")
        print("="*60)
        print(f"Block size: {self.block_size}")
        print(f"Mask token ID: {self.mask_id}")
        print(f"Total masked tokens: {torch.sum(mask_index).item()}")
        print(f"Start position: {start}")
        print()

        for iteration in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            num_masked = torch.sum(mask_index).item()

            if num_masked == 0:
                print(f"Iteration {iteration}: All tokens unmasked!")
                break

            print(f"Iteration {iteration}:")
            print(f"  Masked count: {num_masked}")

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
                first_masked_idx = masked_positions[0].item()
                transfer_index[first_masked_idx] = True

                # Track decoding order
                self.decoding_order.append(first_masked_idx)

                # Get predicted token
                predicted_token = x[first_masked_idx].item()

                print(f"  Selected position: {first_masked_idx}")
                print(f"  Predicted token ID: {predicted_token}")
                print(f"  Remaining positions: {masked_positions[1:].tolist()}")

            # Unmask the selected token
            forward_batch.input_ids[transfer_index] = x[transfer_index]

            # Show current state
            current_state = forward_batch.input_ids.tolist()
            masked_display = ['M' if id == self.mask_id else str(id) for id in current_state]
            print(f"  Current state: {masked_display[-self.block_size:]}")
            print()

        # Final forward pass with all tokens unmasked
        print("Final forward pass with all tokens unmasked...")
        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        next_token_ids = forward_batch.input_ids[start:]

        print("\n" + "="*60)
        print("DECODING SUMMARY")
        print("="*60)
        print(f"Decoding order: {self.decoding_order}")
        print(f"Number of iterations: {len(self.decoding_order)}")
        print(f"Generated tokens: {next_token_ids.tolist()}")
        print("="*60 + "\n")

        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = SequentialVerbose
