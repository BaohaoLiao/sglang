from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidenceVerbose(DllmAlgorithm):
    """LowConfidence algorithm with verbose output.

    Unmasks tokens based on confidence scores (highest confidence first).
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
        self.confidence_scores = []

        print("\n" + "="*60)
        print("LOW CONFIDENCE DLLM DECODING - Verbose Mode")
        print("="*60)
        print(f"Block size: {self.block_size}")
        print(f"Mask token ID: {self.mask_id}")
        print(f"Total masked tokens: {torch.sum(mask_index).item()}")
        print(f"Start position: {start}")
        print(f"Strategy: Unmask HIGHEST confidence token first")
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

            # Get predicted tokens and their probabilities
            x = torch.argmax(logits_output.full_logits, dim=-1)
            p = torch.squeeze(
                torch.gather(
                    F.softmax(logits_output.full_logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )

            # Only consider masked positions for confidence
            x = torch.where(mask_index, x, forward_batch.input_ids)
            confidence = torch.where(mask_index, p, torch.tensor(-np.inf))

            # Get confidence scores for all masked positions (for display)
            masked_positions = torch.where(mask_index)[0]
            masked_confidences = []
            for pos in masked_positions:
                conf = confidence[pos].item()
                if conf != -np.inf:
                    masked_confidences.append((pos.item(), conf))

            # Sort by confidence for display
            masked_confidences.sort(key=lambda x: x[1], reverse=True)

            print(f"  Confidence scores (masked positions):")
            for pos, conf in masked_confidences:
                print(f"    Position {pos}: {conf:.4f}")

            # Select highest confidence masked token
            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            _, select_index = torch.topk(confidence, k=1)
            transfer_index[select_index] = True

            selected_pos = select_index[0].item()
            selected_conf = confidence[selected_pos].item()
            predicted_token = x[selected_pos].item()

            # Track decoding order
            self.decoding_order.append(selected_pos)
            self.confidence_scores.append(selected_conf)

            print(f"  → Selected position: {selected_pos} (confidence: {selected_conf:.4f})")
            print(f"  → Predicted token ID: {predicted_token}")

            # Unmask the selected token
            forward_batch.input_ids[transfer_index] = x[transfer_index]

            # Show current state
            current_state = forward_batch.input_ids.tolist()
            masked_display = ['M' if id == self.mask_id else str(id) for id in current_state]
            print(f"  → Current state: {masked_display[-self.block_size:]}")
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
        print(f"Confidence scores: {[f'{c:.4f}' for c in self.confidence_scores]}")
        print(f"Number of iterations: {len(self.decoding_order)}")
        print(f"Generated tokens: {next_token_ids.tolist()}")
        print()
        print("Analysis:")
        print(f"  - Highest confidence: {max(self.confidence_scores):.4f} (position {self.decoding_order[self.confidence_scores.index(max(self.confidence_scores))]})")
        print(f"  - Lowest confidence: {min(self.confidence_scores):.4f} (position {self.decoding_order[self.confidence_scores.index(min(self.confidence_scores))]})")
        print(f"  - Average confidence: {np.mean(self.confidence_scores):.4f}")
        print("="*60 + "\n")

        return logits_output, next_token_ids, can_run_cuda_graph


Algorithm = LowConfidenceVerbose
