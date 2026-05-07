import torch
from PIL import Image as PILImage
from outlines import from_transformers, Generator, inputs
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.schemas import make_response_schema


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level cache so the (expensive) JSON-schema logits processor is built
# once and reused across all evaluation samples.
_generator_cache: dict[tuple[int, int], Generator] = {}


def load_model_and_processor():
    """Load Qwen3-VL model and processor locally."""
    print(f"Loading model {MODEL_ID} on {DEVICE} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def _get_cached_generator(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    num_options: int,
) -> Generator:
    """Return a cached Outlines Generator for the given model/processor pair.

    Generators are cached by ``(model_id, num_options)`` so that the
    constrained decoder only allows the exact set of option letters that
    exist in the current MMMU sample.
    """
    key = (id(model), num_options)
    if key not in _generator_cache:
        outlines_model = from_transformers(model, processor)
        output_schema = make_response_schema(num_options)
        _generator_cache[key] = Generator(outlines_model, output_type=output_schema)
    return _generator_cache[key]


def run_inference(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    images: list[PILImage.Image],
    prompt: str,
    num_options: int = 4,
    max_new_tokens: int = 2048,
) -> str:
    """
    Run structured generation via Outlines + Pydantic schema and return
    the raw JSON string produced by the model.
    """
    generator = _get_cached_generator(model, processor, num_options)

    # Build a chat prompt with a system instruction that tells the model to
    # emit only valid JSON matching the Pydantic schema.
    chat = inputs.Chat()
    chat.add_system_message(
        "You are a helpful assistant. For multiple-choice questions, think "
        "step by step and explain your reasoning, then respond with ONLY "
        "valid JSON matching the provided schema. Do not include markdown, "
        "explanations, or any text outside the JSON."
    )

    # Attach images (if any) via outlines' multimodal inputs helper.
    content_parts = [prompt]
    for img in images:
        content_parts.append(inputs.Image(img))

    chat.add_user_message(content_parts)

    # The logits processor enforces valid JSON that conforms to ModelResponse
    # at every decoding step, effectively preventing run-away reasoning chains.
    raw_answer = generator(
        chat,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    return raw_answer.strip()
