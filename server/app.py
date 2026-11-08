"""FastAPI server for nanoserve with OpenAI-compatible API and SSE streaming."""

import asyncio
import json
import time
from typing import AsyncGenerator, List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nanoserve.engine import LLMEngine
from nanoserve.types import SamplingParams

app = FastAPI(
    title="nanoserve",
    description="A lightweight LLM inference engine with Paged KV Cache and Continuous Batching",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine singleton
engine: Optional[LLMEngine] = None


@app.on_event("startup")
def startup_event():
    global engine
    engine = LLMEngine(
        model_name_or_path="HuggingFaceTB/SmolLM2-135M-Instruct",
        num_blocks=256,
        block_size=16,
        max_batch_size=8,
    )


class CompletionRequest(BaseModel):
    model: Optional[str] = "SmolLM2-135M-Instruct"
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Optional[str] = "length"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: dict


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "engine": "nanoserve",
        "model": engine.model.config.vocab_size if engine else "loading",
    }


@app.get("/stats")
def stats():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    return {
        "num_free_blocks": engine.allocator.num_free_blocks,
        "num_allocated_blocks": engine.allocator.num_allocated_blocks,
        "total_blocks": engine.allocator.num_blocks,
        "block_size": engine.allocator.block_size,
        "active_running": len(engine.scheduler.running),
        "waiting": len(engine.scheduler.waiting),
        "total_preemptions": engine.scheduler.num_preemptions,
    }


async def stream_generator(req_id: str, prompt_text: str, params: SamplingParams) -> AsyncGenerator[str, None]:
    """Stream token deltas formatted as SSE chunks."""
    req = engine.add_request(req_id, prompt_text, params)
    last_idx = 0

    while not req.is_finished() or last_idx < len(req.output_token_ids):
        # Step engine
        engine.step()

        # Emit any new tokens
        while last_idx < len(req.output_token_ids):
            token_id = req.output_token_ids[last_idx]
            token_str = engine.tokenizer.decode([token_id])
            last_idx += 1

            chunk = {
                "id": req_id,
                "object": "text_completion",
                "created": int(time.time()),
                "choices": [{"text": token_str, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        await asyncio.sleep(0.001)

    # Final EOS chunk
    final_chunk = {
        "id": req_id,
        "object": "text_completion",
        "created": int(time.time()),
        "choices": [{"text": "", "index": 0, "finish_reason": req.state.value}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def create_completion(req: CompletionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    prompt_str = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    req_id = f"cmpl_{int(time.time()*1000)}"
    params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_new_tokens=req.max_tokens,
    )

    if req.stream:
        return StreamingResponse(
            stream_generator(req_id, prompt_str, params),
            media_type="text/event-stream",
        )

    # Non-streaming
    outputs = engine.generate([prompt_str], sampling_params=params)
    out = outputs[0]

    return CompletionResponse(
        id=req_id,
        created=int(time.time()),
        model="SmolLM2-135M-Instruct",
        choices=[
            CompletionChoice(
                text=out.output_text,
                index=0,
                finish_reason=out.finish_reason,
            )
        ],
        usage={
            "prompt_tokens": len(out.prompt_token_ids),
            "completion_tokens": len(out.output_token_ids),
            "total_tokens": len(out.prompt_token_ids) + len(out.output_token_ids),
            "ttft_ms": round(out.ttft_ms, 2),
            "itl_ms": round(out.inter_token_latency_ms, 2),
            "total_time_ms": round(out.total_time_ms, 2),
        },
    )

