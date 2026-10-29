"""Continuous batching scheduler: waiting and running queues, admission control, and preemption."""

import time
from dataclasses import dataclass
from typing import List, Optional
from nanoserve.types import Request, SequenceState
from nanoserve.kv_cache import BlockAllocator


@dataclass
class ScheduleResult:
    """Decisions made by the scheduler for a single step."""
    prefills: List[Request]
    decodes: List[Request]
    preempted: List[Request]


class Scheduler:
    """Continuous batching scheduler with demand-paged memory and recompute preemption."""

    def __init__(
        self,
        allocator: BlockAllocator,
        max_batch_size: int = 16,
        max_num_batched_tokens: int = 512,
    ):
        self.allocator = allocator
        self.max_batch_size = max_batch_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = allocator.block_size

        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.num_preemptions: int = 0

    def add_request(self, request: Request):
        """Add incoming request to waiting queue."""
        request.state = SequenceState.WAITING
        self.waiting.append(request)

    def abort_request(self, request_id: str):
        """Abort request and free its blocks immediately."""
        for i, req in enumerate(self.waiting):
            if req.request_id == request_id:
                self.waiting.pop(i)
                return

        for i, req in enumerate(self.running):
            if req.request_id == request_id:
                self.allocator.free_table(req.block_table)
                self.running.pop(i)
                return

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> ScheduleResult:
        """Schedule next step: handle running decode requests and admit new waiting requests."""
        # 1. Clean up finished requests from running queue and free their memory blocks
        finished = [req for req in self.running if req.is_finished()]
        for req in finished:
            self.allocator.free_table(req.block_table)
            self.running.remove(req)

        # 2. Check running requests: ensure each has a block allocated for its next decode token
        preempted: List[Request] = []
        # Process in reverse order so youngest requests are preempted first if memory is exhausted
        for req in list(reversed(self.running)):
            needed_blocks = (req.num_total_tokens + 1 + self.block_size - 1) // self.block_size
            if len(req.block_table) < needed_blocks:
                if self.allocator.num_free_blocks < 1:
                    # Preemption: free youngest request's blocks and return to waiting queue
                    preempt_req = self.running.pop()
                    self.allocator.free_table(preempt_req.block_table)
                    preempt_req.state = SequenceState.PREEMPTED
                    preempt_req.output_token_ids.clear()  # reset tokens for recomputation
                    self.waiting.insert(0, preempt_req)   # high priority on front of waiting queue
                    preempted.append(preempt_req)
                    self.num_preemptions += 1
                else:
                    req.block_table.append(self.allocator.allocate())

        decodes = list(self.running)

        # 3. Admit new requests from waiting queue if batch size and memory allow
        prefills: List[Request] = []
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting[0]
            prompt_len = len(req.prompt_token_ids)
            needed_blocks = (prompt_len + self.block_size - 1) // self.block_size

            # Check if allocator has enough blocks
            if self.allocator.num_free_blocks < needed_blocks:
                break  # Not enough free blocks, wait until running requests finish

            # Admit request
            self.waiting.pop(0)
            req.state = SequenceState.RUNNING
            if req.start_time is None:
                req.start_time = time.time()
            req.block_table = [self.allocator.allocate() for _ in range(needed_blocks)]
            self.running.append(req)
            prefills.append(req)

        return ScheduleResult(prefills=prefills, decodes=decodes, preempted=preempted)

