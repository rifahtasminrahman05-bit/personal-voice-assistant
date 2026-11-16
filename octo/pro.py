# -*- coding: utf-8 -*-
"""Octo-Pro — fine-tune a pretrained Hugging Face model with LoRA (optional, GPU recommended).

This is the "expert mode" path: a 0.5B-1.1B pretrained model + QLoRA fine-tune on
your own Q&A data gives vastly better coding/security answers than the tiny from-scratch
Octo. Requires: pip install transformers peft datasets bitsandbytes accelerate.

Prepare data as JSONL: {"prompt": "...", "response": "..."} per line.

Usage (GPU machine):
    python -m octo.pro --train --data data/octo/pro_train.jsonl
    python -m octo.pro --chat   # interactive with the fine-tuned model
"""
import argparse
import json
import os

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "octo_checkpoints/octo-pro-adapter"


def _require(*pkgs):
    missing = [p for p in pkgs if __import__("importlib").util.find_spec(p) is None]
    if missing:
        raise SystemExit(f"missing dependencies: {', '.join(missing)}\n"
                         "install with: pip install transformers peft datasets accelerate")


def train(data_path, base_model=BASE_MODEL, epochs=3, lr=2e-4):
    _require("transformers", "peft", "datasets", "accelerate")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer, TrainingArguments)

    samples = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    texts = [f"### Instruction:\n{s['prompt']}\n### Response:\n{s['response']}" for s in samples]
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=512), batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(model=model, train_dataset=ds, data_collator=collator,
                      args=TrainingArguments(
                          output_dir=ADAPTER_DIR, num_train_epochs=epochs, learning_rate=lr,
                          per_device_train_batch_size=4, gradient_accumulation_steps=4,
                          logging_steps=10, save_strategy="epoch", report_to=[]))
    trainer.train()
    model.save_pretrained(ADAPTER_DIR)
    print(f"adapter saved: {ADAPTER_DIR}")


def chat(prompt, adapter=ADAPTER_DIR):
    _require("transformers", "peft")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto",
                                                 torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter)
    inputs = tokenizer(f"### Instruction:\n{prompt}\n### Response:\n", return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7)
    print(tokenizer.decode(out[0], skip_special_tokens=True))


def main():
    ap = argparse.ArgumentParser(description="Octo-Pro: LoRA fine-tune of a pretrained LLM.")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--data", default="data/octo/pro_train.jsonl")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--base-model", default=BASE_MODEL)
    args = ap.parse_args()

    if args.train:
        if not os.path.exists(args.data):
            raise SystemExit(f"create {args.data} first: one JSON object per line "
                             '{"prompt": "...", "response": "..."}')
        train(args.data, base_model=args.base_model)
    elif args.prompt:
        chat(args.prompt)
    else:
        ap.error("pass --train or --prompt")


if __name__ == "__main__":
    main()
