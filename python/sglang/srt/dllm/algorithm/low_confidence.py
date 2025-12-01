from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


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
        # NOTE: DLLM currently assumes batch size 1.
        assert (
            forward_batch.input_ids.dim() == 2 and forward_batch.input_ids.size(0) == 1
        ), "DLLM currently supports batch size 1"
        input_ids = forward_batch.input_ids  # shape [1, seq]
        mask_index = input_ids == self.mask_id
        mask_positions = torch.nonzero(mask_index[0], as_tuple=False).flatten()
        if mask_positions.numel() == 0:
            # No masks detected; return a normal forward pass with empty order.
            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )
            return logits_output, torch.empty(0, device=input_ids.device), can_run_cuda_graph, torch.empty(0, device=input_ids.device, dtype=torch.int32)

        start = mask_positions.min().item()
        decoding_order = []

        for _ in range(self.block_size):
            mask_index = input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )

            x = torch.argmax(logits_output.full_logits, dim=-1)  # [1, seq]
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
            forward_batch.input_ids = input_ids

        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        next_token_ids = input_ids[0, start:]
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
