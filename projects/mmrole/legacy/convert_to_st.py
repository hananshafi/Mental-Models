from transformers import AutoModelForCausalLM

src = "Qwen/Qwen-VL-Chat"
dst = "artifacts/huggingface/hub/Qwen-VL-Chat-safetensors"

model = AutoModelForCausalLM.from_pretrained(
    src,
    trust_remote_code=True,
    device_map="cpu",
)

model.save_pretrained(dst, safe_serialization=True)
