"""LLMEngine: High-level inference engine coordinating model, scheduler, KV cache, and sampler."""

import time
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer

from nanoserve.types import Request, RequestOutput, SamplingParams
from nanoserve.model import NanoserveModel
from nanoserve.kv_cache import BlockAllocator, PagedKVCache
from nanoserve.scheduler import Scheduler
from nanoserve.sampler import Sampler


class LLMEngine:
    """Continuous batching LLM inference engine."""

    def __init__(
        self,
        model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
        num_blocks: int = 128,
        block_size: int = 16,
        max_batch_size: int = 16,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = NanoserveModel.from_pretrained(model_name_or_path, device=device, dtype=dtype)
        self.model.eval()

        self.eos_token_id = self.model.config.eos_token_id
        self.allocator = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
        self.paged_cache = PagedKVCache(
            num_layers=self.model.config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=self.model.config.num_key_value_heads,
            head_dim=self.model.config.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self.scheduler = Scheduler(
            allocator=self.allocator,
            max_batch_size=max_batch_size,
        )
        self.sampler = Sampler()

        # Track completed requests for lookup
        self.finished_requests: Dict[str, Request] = {}
        self.all_requests: Dict[str, Request] = {}

    def add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Request:
        """Add request to engine."""
        input_ids = self.tokenizer.encode(prompt)
        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=input_ids,
            sampling_params=sampling_params or SamplingParams(),
            arrival_time=time.time(),
        )
        self.scheduler.add_request(request)
        self.all_requests[request_id] = request
        return request

    def has_unfinished_requests(self) -> bool:
        """Check if any requests remain in waiting or running state."""
        return self.scheduler.has_unfinished_requests()

    def step(self) -> List[RequestOutput]:
        """Execute one continuous batching step."""
        schedule_result = self.scheduler.schedule()
        updated_requests: List[Request] = []

        # 1. Process Prefills (newly admitted requests)
        for req in schedule_result.prefills:
            x = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=self.device)
            prompt_len = len(req.prompt_token_ids)

            with torch.no_grad():
                logits = self.model(
                    x,
                    paged_cache=self.paged_cache,
                    block_tables=[req.block_table],
                    seq_lens=[prompt_len],
                )

            # Sample next token from last position of prompt logits
            last_logits = logits[:, -1, :]  # [1, vocab_size]
            tokens = self.sampler.sample(last_logits, [req.sampling_params])
            next_token = tokens[0]
            req.append_token(next_token, eos_token_id=self.eos_token_id)
            updated_requests.append(req)

        # 2. Process Decodes (batched continuous decode across all running sequences)
        # Note: requests that just underwent prefill are already stepped, so we only decode
        # requests that were already running before prefill
        decode_candidates = [
            req for req in schedule_result.decodes
            if req not in schedule_result.prefills and not req.is_finished()
        ]

        if decode_candidates:
            B = len(decode_candidates)
            input_tokens = [[req.output_token_ids[-1]] for req in decode_candidates]
            x_step = torch.tensor(input_tokens, dtype=torch.long, device=self.device)
            block_tables = [req.block_table for req in decode_candidates]
            seq_lens = [req.num_total_tokens for req in decode_candidates]

            with torch.no_grad():
                logits = self.model(
                    x_step,
                    paged_cache=self.paged_cache,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                )

            # Logits shape: [B, 1, vocab_size] -> squeeze to [B, vocab_size]
            decode_logits = logits.squeeze(1)
            sampling_params_list = [req.sampling_params for req in decode_candidates]
            sampled_tokens = self.sampler.sample(decode_logits, sampling_params_list)

            for i, req in enumerate(decode_candidates):
                req.append_token(sampled_tokens[i], eos_token_id=self.eos_token_id)
                updated_requests.append(req)

        # Convert updated requests to RequestOutput
        outputs: List[RequestOutput] = []
        for req in updated_requests:
            now = time.time()
            ttft = ((req.first_token_time - req.arrival_time) * 1000.0) if req.first_token_time else 0.0
            total_time = ((now - req.arrival_time) * 1000.0)
            num_tokens = len(req.output_token_ids)
            itl = (total_time - ttft) / max(1, num_tokens - 1) if num_tokens > 1 else 0.0

            text = self.tokenizer.decode(req.output_token_ids, skip_special_tokens=True)
            output = RequestOutput(
                request_id=req.request_id,
                prompt=req.prompt,
                prompt_token_ids=req.prompt_token_ids,
                output_token_ids=list(req.output_token_ids),
                output_text=text,
                finished=req.is_finished(),
                finish_reason=req.state.value if req.is_finished() else None,
                ttft_ms=ttft,
                inter_token_latency_ms=itl,
                total_time_ms=total_time,
            )
            outputs.append(output)

        return outputs

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """Convenience method to execute a batch of prompts to completion."""
        params = sampling_params or SamplingParams()
        req_ids = []
        for i, p in enumerate(prompts):
            req_id = f"req_{i}_{int(time.time()*1000)}"
            self.add_request(req_id, p, params)
            req_ids.append(req_id)

        while self.has_unfinished_requests():
            self.step()

        # Gather final outputs
        results: List[RequestOutput] = []
        for req_id in req_ids:
            req = self.all_requests[req_id]
            ttft = ((req.first_token_time - req.arrival_time) * 1000.0) if req.first_token_time else 0.0
            total_time = ((req.finish_time - req.arrival_time) * 1000.0) if req.finish_time else 0.0
            num_tokens = len(req.output_token_ids)
            itl = (total_time - ttft) / max(1, num_tokens - 1) if num_tokens > 1 else 0.0

            text = self.tokenizer.decode(req.output_token_ids, skip_special_tokens=True)
            results.append(
                RequestOutput(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    prompt_token_ids=req.prompt_token_ids,
                    output_token_ids=list(req.output_token_ids),
                    output_text=text,
                    finished=req.is_finished(),
                    finish_reason=req.state.value,
                    ttft_ms=ttft,
                    inter_token_latency_ms=itl,
                    total_time_ms=total_time,
                )
            )

        return results

