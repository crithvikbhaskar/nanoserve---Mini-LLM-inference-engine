"""Token sampler supporting greedy, temperature, top-k, and top-p sampling."""

from typing import List, Optional
import torch
from nanoserve.types import SamplingParams


class Sampler:
    """Samples next token from logits given per-request sampling parameters."""

    def sample(
        self,
        logits: torch.Tensor,
        sampling_params: List[SamplingParams],
    ) -> List[int]:
        """Sample next token for each row in logits.
        
        Args:
            logits: Float tensor of shape [batch_size, vocab_size]
            sampling_params: List of SamplingParams per request of length batch_size
        Returns:
            List of selected token IDs
        """
        batch_size = logits.shape[0]
        assert len(sampling_params) == batch_size, "Batch size mismatch"

        selected_tokens: List[int] = []

        for i in range(batch_size):
            params = sampling_params[i]
            row_logits = logits[i].float()  # work in fp32 for numerical stability

            # Greedy path
            if params.temperature <= 1e-5:
                next_token = int(torch.argmax(row_logits).item())
                selected_tokens.append(next_token)
                continue

            # Temperature scaling
            row_logits = row_logits / params.temperature

            # Top-K filtering
            if params.top_k > 0:
                top_k = min(params.top_k, row_logits.size(-1))
                values, _ = torch.topk(row_logits, top_k)
                min_value = values[-1]
                row_logits = torch.where(
                    row_logits < min_value,
                    torch.tensor(float("-inf"), device=row_logits.device),
                    row_logits,
                )

            # Top-P (nucleus) filtering
            if params.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(row_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > params.top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                row_logits[indices_to_remove] = float("-inf")

            # Softmax to probabilities
            probs = torch.softmax(row_logits, dim=-1)

            # Seeded generator if requested
            generator: Optional[torch.Generator] = None
            if params.seed is not None:
                generator = torch.Generator(device=probs.device)
                generator.manual_seed(params.seed)

            # Categorical sampling
            next_token = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
            selected_tokens.append(next_token)

        return selected_tokens

