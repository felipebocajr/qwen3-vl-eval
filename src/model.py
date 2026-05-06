import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_processor(model_id: str = MODEL_ID):
    """Load Qwen3-VL model and processor locally."""
    print(f"Loading model {model_id} on {DEVICE} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def run_inference(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    images: list[Image.Image],
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    """Run a single forward pass and return the decoded response."""
    messages = [
        {
            "role": "user",
            "content": [
                * [{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    raw_answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return raw_answer
