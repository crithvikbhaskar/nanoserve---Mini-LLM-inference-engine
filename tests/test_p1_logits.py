import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanoserve.model import NanoserveModel

def test_logits_match():
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer("The capital of France is", return_tensors="pt")

    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    hf_model.eval()

    my_model = NanoserveModel.from_pretrained(model_id, dtype=torch.float32)
    my_model.eval()

    with torch.no_grad():
        hf_logits = hf_model(**inputs).logits
        my_logits = my_model(inputs.input_ids)

    diff = (hf_logits - my_logits).abs().max().item()
    print(f"Max absolute logit diff: {diff:.6e}")
    assert diff < 1e-3, f"Logit difference too high: {diff}"

if __name__ == "__main__":
    test_logits_match()

