"""Type definitions and data structures for nanoserve."""

from enum import Enum
import time
from typing import List, Optional
from pydantic import BaseModel, Field


class SequenceState(str, Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    FINISHED_STOP = "FINISHED_STOP"
    FINISHED_LENGTH = "FINISHED_LENGTH"


class SamplingParams(BaseModel):
    temperature: float = Field(default=1.0, ge=0.0)
    top_k: int = Field(default=-1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=64, ge=1)
    ignore_eos: bool = Field(default=False)
    seed: Optional[int] = Field(default=None)


class Request:
    """Internal state of a sequence being processed by the engine."""

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        sampling_params: Optional[SamplingParams] = None,
        arrival_time: Optional[float] = None,
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = sampling_params or SamplingParams()
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.output_token_ids: List[int] = []
        self.state = SequenceState.WAITING

        # Timestamps for metrics
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None

        # Paged cache management
        self.block_table: List[int] = []
        self.num_computed_tokens: int = 0

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def is_finished(self) -> bool:
        return self.state in (SequenceState.FINISHED_STOP, SequenceState.FINISHED_LENGTH)

    def append_token(self, token_id: int, eos_token_id: Optional[int] = None):
        self.output_token_ids.append(token_id)
        now = time.time()
        if self.first_token_time is None:
            self.first_token_time = now

        if (
            not self.sampling_params.ignore_eos
            and eos_token_id is not None
            and token_id == eos_token_id
        ):
            self.state = SequenceState.FINISHED_STOP
            self.finish_time = now
        elif len(self.output_token_ids) >= self.sampling_params.max_new_tokens:
            self.state = SequenceState.FINISHED_LENGTH
            self.finish_time = now


class RequestOutput(BaseModel):
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    output_token_ids: List[int]
    output_text: str = ""
    finished: bool
    finish_reason: Optional[str] = None
    ttft_ms: float = 0.0
    inter_token_latency_ms: float = 0.0
    total_time_ms: float = 0.0

