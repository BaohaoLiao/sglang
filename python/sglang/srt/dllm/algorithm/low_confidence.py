from typing import Optional, Tuple, Union

import logging
import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class LowConfidence(DllmAlgorithm):

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor],
        Optional[torch.Tensor],
        bool,
        torch.Tensor,
    ]:
        input_ids = forward_batch.input_ids
        # Flatten to 1D (seq) for DLLM processing; we only support bs=1.
        if input_ids.dim() == 2:
            if input_ids.size(0) != 1:
                raise ValueError(
                    f"DLLM currently supports batch size 1, got shape {tuple(input_ids.shape)}"
                )
            input_ids = input_ids[0]
        elif input_ids.dim() != 1:
            raise ValueError(
                f"DLLM currently supports 1D or [1, seq] input_ids, got shape {tuple(input_ids.shape)}"
            )
        mask_index = input_ids == self.mask_id
        mask_positions = torch.nonzero(mask_index, as_tuple=False).flatten()
        if mask_positions.numel() == 0:
            # Fallback: treat the last block as masks and overwrite them.
            seq_len = input_ids.size(0)
            start = max(seq_len - self.block_size, 0)
            mask_positions = torch.arange(
                start, seq_len, device=input_ids.device, dtype=torch.int64
            )
            input_ids[mask_positions] = self.mask_id
            mask_index = input_ids == self.mask_id
        else:
            start = mask_positions.min().item()
        decoding_order = []

        # Print to stdout for visibility even when logging is filtered.
        print(
            f"[DLLM LowConfidence] start={start} mask_positions={mask_positions.tolist()} "
            f"input_tail={input_ids[-self.block_size :].tolist()}",
            flush=True,
        )

        for step in range(self.block_size):
            mask_index = input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            x = torch.argmax(logits_output.full_logits, dim=-1)  # [seq]
            p = torch.squeeze(
                torch.gather(
                    F.softmax(logits_output.full_logits, dim=-1),
                    dim=-1,
                    index=torch.unsqueeze(x, -1),
                ),
                -1,
            )
            x = torch.where(mask_index, x, input_ids)
            confidence = torch.where(mask_index, p, -np.inf)
            transfer_index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            _, select_index = torch.topk(confidence, k=1, dim=-1)
            transfer_index.scatter_(-1, select_index, True)

            decoding_order.append(int(select_index.item() - start))
            input_ids = torch.where(transfer_index, x, input_ids)
            forward_batch.input_ids = input_ids.unsqueeze(0)

            print(
                f"[DLLM LowConfidence] step={step} select_index={select_index.item()} "
                f"decoding_order={decoding_order} remaining_masks={int(torch.sum(input_ids == self.mask_id))}",
                flush=True,
            )

        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        next_token_ids = input_ids[start:]
        # Emit an info-level log so it shows up without DEBUG handlers.
        logger.info(
            "DLLM LowConfidence run: start=%s mask_positions=%s decoding_order=%s next_token_ids=%s",
            start,
            mask_positions.tolist(),
            decoding_order,
            next_token_ids.tolist(),
        )
        return (
            logits_output,
            next_token_ids,
            can_run_cuda_graph,
            torch.tensor(
                decoding_order,
                device=forward_batch.input_ids.device,
                dtype=torch.int32,
            ),
        )


Algorithm = LowConfidence
