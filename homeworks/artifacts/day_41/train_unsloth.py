#!/usr/bin/env python3
"""Local LoRA/QLoRA SFT for day_41 aviation JSONL via Unsloth.

Apple Silicon (MLX): Unsloth patches ``trl.SFTTrainer`` / ``trl.SFTConfig``.
Import ``unsloth`` BEFORE ``trl`` trainers, otherwise you get:
  AttributeError: '_MLXSFTConfig' object has no attribute 'model_init_kwargs'

Requires a separate env, e.g.:
  python3 -m venv .venv-unsloth && source .venv-unsloth/bin/activate
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install datasets transformers trl peft accelerate bitsandbytes

Train:
  python homeworks/artifacts/day_41/train_unsloth.py \\
    --train homeworks/artifacts/day_41/train.jsonl \\
    --eval homeworks/artifacts/day_41/eval.jsonl \\
    --out homeworks/artifacts/day_41/unsloth_out

Export GGUF (after train):
  python homeworks/artifacts/day_41/train_unsloth.py --export-gguf \\
    --out homeworks/artifacts/day_41/unsloth_out \\
    --gguf-out homeworks/artifacts/day_41/aviation-faa-q4_k_m.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def formatting_prompts_func(examples: dict, tokenizer) -> dict:
    texts: list[str] = []
    for messages in examples["messages"]:
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return {"text": texts}


def _is_mlx_trl_shim() -> bool:
    import trl

    return bool(getattr(trl, "__UNSLOTH_MLX_COMPAT__", False))


def _sft_config(
    *,
    output_dir: str,
    batch_size: int,
    max_steps: int,
    max_seq_length: int,
):
    """Build SFTConfig for CUDA TRL or Unsloth MLX shim."""
    from trl import SFTConfig

    # MLX maps adamw_* → adamw; CUDA can use adamw_8bit.
    if sys.platform == "darwin" or _is_mlx_trl_shim():
        optim = "adamw_8bit"  # accepted alias → adamw on MLX
    else:
        optim = "adamw_8bit"

    kwargs: dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 5,
        "max_steps": max_steps,
        "learning_rate": 2e-4,
        "logging_steps": 5,
        "optim": optim,
        "seed": 41,
        "report_to": "none",
        "dataset_text_field": "text",
        # Pass both names: TRL 1.x wants max_length; Unsloth MLX prefers max_seq_length.
        "max_length": max_seq_length,
        "max_seq_length": max_seq_length,
    }
    return SFTConfig(**kwargs)


def train(args: argparse.Namespace) -> None:
    # CRITICAL on Apple Silicon: import unsloth first so it patches trl.SFTTrainer
    # / SFTConfig to the MLX-backed shims. Importing real TRL SFTTrainer first and
    # then using MLX SFTConfig causes model_init_kwargs AttributeError.
    from unsloth import FastLanguageModel
    from trl import SFTTrainer

    from datasets import Dataset

    print(
        f"TRL backend: {'Unsloth MLX shim' if _is_mlx_trl_shim() else 'stock TRL'}",
        flush=True,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=41,
    )

    train_rows = load_jsonl(args.train)
    eval_rows = load_jsonl(args.eval) if args.eval and args.eval.is_file() else []
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows) if eval_rows else None

    def _fmt(batch: dict) -> dict:
        return formatting_prompts_func(batch, tokenizer)

    train_ds = train_ds.map(_fmt, batched=True)
    if eval_ds is not None:
        eval_ds = eval_ds.map(_fmt, batched=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sft_args = _sft_config(
        output_dir=str(out),
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
    )

    trainer_kwargs: dict = {
        "model": model,
        "train_dataset": train_ds,
        "args": sft_args,
        "tokenizer": tokenizer,
    }
    # MLX UnslothTrainer accepts tokenizer; stock TRL 1.x wants processing_class.
    if not _is_mlx_trl_shim():
        trainer_kwargs.pop("tokenizer", None)
        trainer_kwargs["processing_class"] = tokenizer
        if eval_ds is not None:
            trainer_kwargs["eval_dataset"] = eval_ds
    else:
        # Keep eval optional; some MLX paths ignore it safely if passed.
        if eval_ds is not None:
            trainer_kwargs["eval_dataset"] = eval_ds

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(str(out / "lora"))
    tokenizer.save_pretrained(str(out / "lora"))
    print(f"Saved LoRA adapter to {out / 'lora'}")


def export_gguf(args: argparse.Namespace) -> None:
    from unsloth import FastLanguageModel

    out = Path(args.out)
    lora_dir = out / "lora"
    if not lora_dir.is_dir():
        raise SystemExit(f"LoRA dir not found: {lora_dir} (train first)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_dir),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    gguf = Path(args.gguf_out)
    gguf.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(gguf.parent / "gguf_export"),
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(
        f"GGUF export folder: {gguf.parent / 'gguf_export'}\n"
        f"Copy/rename the .gguf file to {gguf} and update Modelfile.aviation-faa FROM path."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="day_41 Unsloth train / export")
    parser.add_argument("--train", type=Path, default=Path("homeworks/artifacts/day_41/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("homeworks/artifacts/day_41/eval.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("homeworks/artifacts/day_41/unsloth_out"))
    parser.add_argument("--model", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--export-gguf", action="store_true")
    parser.add_argument(
        "--gguf-out",
        type=Path,
        default=Path("homeworks/artifacts/day_41/aviation-faa-q4_k_m.gguf"),
    )
    args = parser.parse_args()

    if args.export_gguf:
        export_gguf(args)
    else:
        if not args.train.is_file():
            raise SystemExit(f"train file not found: {args.train}")
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
