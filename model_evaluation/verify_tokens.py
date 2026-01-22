import os
import sys
from model_evaluation.config import Settings
from transformers import AutoTokenizer


def verify_token_counting():
    settings = Settings()
    # Use 1b/2b tokenizer (gemma-2-2b-it is open weights usually or gated? Gemma 3 is gated)
    # I'll use the ID from settings or hardcode a gemma identifier if available.
    # We'll try to load the tokenizer used in the project.

    model_id = "google/gemma-2-2b-it"  # Use a known gemma tokenizer
    # Or better, use the logic from module
    try:
        from model_evaluation.main_agent.gemma_model_loader import get_gemma_model_id

        model_id = get_gemma_model_id(size="1b", model_type="it")  # 4b is default
    except Exception:
        pass

    print(f"Loading tokenizer for {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=settings.hf_token)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    text = "Hello, world!"

    # 1. Check Input Counting
    encoded_ids = tokenizer.encode(text)
    tokenizer_call = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()

    print(f"Text: '{text}'")
    print(f"encode() len: {len(encoded_ids)}")
    print(f"tokenizer() len: {len(tokenizer_call)}")
    print(f"encode() ids: {encoded_ids}")

    if len(encoded_ids) != len(tokenizer_call):
        print("❌ CRITICAL: Input counting mismatch!")
    else:
        print("✅ Input counting consistent.")

    # 2. Check Output Re-encoding Discrepancy
    # Simulare generation
    generated_text = "I am a helpful assistant."
    original_ids = tokenizer.encode(generated_text, add_special_tokens=False)  # Pure tokens
    # Note: encode adding special tokens (BOS) is the default.

    re_encoded_with_defaults = tokenizer.encode(generated_text)

    print(f"\nGenerated text: '{generated_text}'")
    print(f"Target length (no special): {len(original_ids)}")
    print(f"Re-encoded (default): {len(re_encoded_with_defaults)}")
    print(f"Re-encoded ids: {re_encoded_with_defaults}")

    if len(re_encoded_with_defaults) > len(original_ids):
        print("⚠️  Re-encoding adds extra tokens (likely BOS)!")
        print(f"   Diff: {len(re_encoded_with_defaults) - len(original_ids)}")

    # Conclusion
    print("\nCONCLUSION:")
    if len(re_encoded_with_defaults) != len(original_ids):
        print("  Using tokenizer.encode(answer) INFLATES count (adds BOS).")
        print("  Should use len(all_tokens) - prompt_len instead.")
    else:
        print("  Re-encoding seems safe.")


if __name__ == "__main__":
    verify_token_counting()
