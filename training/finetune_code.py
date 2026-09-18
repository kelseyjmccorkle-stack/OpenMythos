#!/usr/bin/env python3
"""
Fine-tune OpenMythos for software development (code) on a streaming code corpus.

This continues an existing OpenMythos checkpoint on a code dataset so the model
specializes for code understanding/generation. It reuses the same FSDP +
grad-accumulation + cosine-schedule machinery as the FineWeb-Edu pretraining
script (`training/3b_fine_web_edu.py`), with fine-tuning defaults: a base
checkpoint to start from, a lower learning rate, a short warmup, and gradient
checkpointing on (recomputes each recurrent-loop iteration in backward to fit
longer code sequences in memory).

Single GPU:
    python training/finetune_code.py --from-checkpoint checkpoints/step_0030000.pt

Multi-GPU:
    torchrun --nproc_per_node=$(python -c "import torch;print(torch.cuda.device_count())") \
        training/finetune_code.py --from-checkpoint checkpoints/step_0030000.pt

The dataset is streamed from the HuggingFace Hub (default: a code corpus with a
"content" text field). Some code datasets are gated — run `huggingface-cli
login` and/or pick a subset with --dataset/--dataset-config/--text-field.

Without --from-checkpoint the model is initialized from scratch on the code
corpus (a warning is logged): that is training-on-code, not fine-tuning, but is
supported for convenience.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from loguru import logger
except ImportError:  # optional dep; fall back to stdlib logging
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("finetune_code")
    logger.success = logger.info  # loguru-only level used below
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from open_mythos import OpenMythos, configure_vocab_size
from open_mythos.main import RecurrentBlock, TransformerBlock
from open_mythos.tokenizer import MythosTokenizer
from open_mythos.variants import mythos_1b


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PackedCodeDataset(IterableDataset):
    """
    Streaming code loader yielding fixed-length (input, target) next-token pairs.

    Identical packing strategy to the pretraining loader: documents are tokenized
    into a rolling buffer and sliced into `seq_len + 1` chunks (input/target
    shifted by one), so every step has the same shape — no pad mask needed.
    Sharding is two-dimensional (`world_size` ranks x DataLoader workers), giving
    each `(rank, worker)` a disjoint slice of the stream with no coordination.

    `datasets` is imported lazily inside `__iter__` so this module (and the core
    training loop) can be imported and unit-tested without the dependency or any
    network access.
    """

    def __init__(
        self,
        encoding,
        seq_len: int,
        dataset: str,
        dataset_config: str | None,
        split: str,
        text_field: str,
        rank: int,
        world_size: int,
    ):
        self.encoding = encoding
        self.seq_len = seq_len
        self.dataset = dataset
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        from datasets import load_dataset

        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        ds = load_dataset(
            self.dataset,
            name=self.dataset_config,
            split=self.split,
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        buf: list[int] = []
        for sample in ds:
            text = sample.get(self.text_field)
            if not text:
                continue
            buf.extend(self.encoding.encode(text))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# LR schedule (linear warmup -> cosine decay), matches the pretraining script
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup to max_lr, then half-cosine decay to min_lr (clamped)."""
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    decay = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# Checkpointing (FSDP-aware; mirrors training/3b_fine_web_edu.py)
# ---------------------------------------------------------------------------


def _list_ckpts(ckpt_dir: str) -> list[str]:
    """Checkpoint paths in ckpt_dir sorted oldest -> newest (zero-padded names)."""
    if not os.path.isdir(ckpt_dir):
        return []
    return sorted(
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.startswith("step_") and f.endswith(".pt")
    )


def save_checkpoint(
    model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master, keep_last=3
) -> None:
    """Gather full model+optimizer state, write atomically, prune old files."""
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not master:
        return

    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg,
            "vocab_size": vocab_size,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)
    for old in _list_ckpts(ckpt_dir)[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            logger.warning(f"Failed to prune old checkpoint {old}: {exc}")
    logger.success(f"Checkpoint saved -> {final_path}")


def load_checkpoint(model, optimizer, path, ddp) -> int:
    """Restore model+optimizer to resume a fine-tuning run; returns the step."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ddp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            model.load_state_dict(ckpt["model"])
            optim_state = FSDP.optim_state_dict_to_load(
                model=model, optim=optimizer, optim_state_dict=ckpt["optimizer"]
            )
            optimizer.load_state_dict(optim_state)
    else:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["step"])


def load_base_checkpoint(path: str):
    """
    Load a base (pretrained) checkpoint for fine-tuning.

    Unlike `load_checkpoint`, this restores only the config + model weights (no
    optimizer state) so fine-tuning starts a fresh optimizer. The model is built
    from the checkpoint's own `cfg`, so the fine-tuned architecture exactly
    matches what was pretrained.

    Returns:
        (model, cfg, base_vocab_size) — model has the pretrained weights loaded.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = OpenMythos(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg, int(ckpt.get("vocab_size", cfg.vocab_size))


# ---------------------------------------------------------------------------
# Core training loop (kept import-friendly so it can be unit-tested on CPU)
# ---------------------------------------------------------------------------


def train_loop(
    model,
    optimizer,
    batch_iter,
    *,
    cfg,
    vocab_size: int,
    device: str,
    total_steps: int,
    start_step: int = 0,
    grad_accum: int = 1,
    warmup_steps: int = 100,
    max_lr: float = 3e-5,
    min_lr: float | None = None,
    global_batch_tok: int | None = None,
    amp_ctx=None,
    ddp: bool = False,
    local_rank: int = 0,
    master: bool = True,
    log_every: int = 10,
    ckpt_every: int = 1000,
    ckpt_dir: str = "checkpoints_code_ft",
    save_ckpts: bool = True,
) -> int:
    """
    Run grad-accumulation training from `start_step` to `total_steps`.

    `batch_iter` is an (effectively infinite) iterator of (input_ids, target_ids)
    long tensors of shape (micro_batch, seq_len). Returns the final step.
    """
    min_lr = max_lr * 0.1 if min_lr is None else min_lr
    amp_ctx = amp_ctx if amp_ctx is not None else nullcontext()
    dst = f"cuda:{local_rank}" if ddp else device
    model.train()
    step = start_step
    t0 = time.perf_counter()

    while step < total_steps:
        cur_lr = get_lr(step, warmup_steps, total_steps, max_lr, min_lr)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum):
            x, y = next(batch_iter)
            x = x.to(dst, non_blocking=True)
            y = y.to(dst, non_blocking=True)
            sync = (
                nullcontext()
                if (not ddp or micro_step == grad_accum - 1)
                else model.no_sync()
            )
            with sync, amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                loss = loss / grad_accum
            loss.backward()
            loss_accum += loss.item()

        if ddp:
            grad_norm = model.clip_grad_norm_(1.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step += 1

        if master and log_every and step % log_every == 0:
            dt = time.perf_counter() - t0
            msg = (
                f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                f"| gnorm {float(grad_norm):.2f} | lr {cur_lr:.2e}"
            )
            if global_batch_tok:
                msg += f" | {global_batch_tok * log_every / dt / 1e6:.2f}M tok/s"
            logger.info(msg)
            t0 = time.perf_counter()

        if save_ckpts and ckpt_every and step % ckpt_every == 0:
            save_checkpoint(
                model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master
            )

    if save_ckpts and step > start_step and (not ckpt_every or step % ckpt_every != 0):
        save_checkpoint(model, optimizer, step, cfg, vocab_size, ckpt_dir, ddp, master)
    return step


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune OpenMythos on code.")
    p.add_argument(
        "--from-checkpoint",
        default=None,
        help="Base pretrained checkpoint (step_*.pt) to fine-tune from. "
        "Omit to train from scratch on code (logs a warning).",
    )
    p.add_argument("--dataset", default="bigcode/the-stack-smol")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-field", default="content")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-5, help="Fine-tuning peak LR.")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--out-dir", default="checkpoints_code_ft")
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable recurrent-loop gradient checkpointing to save memory on "
        "long sequences. NOTE: currently mutually exclusive with the MoE "
        "load-balancing bias update, which is disabled for the run when set.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    master = rank == 0

    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    # --- Model: from a base checkpoint (fine-tune) or from scratch (warns) ---
    if args.from_checkpoint:
        if master:
            logger.info(f"Fine-tuning from base checkpoint: {args.from_checkpoint}")
        model, cfg, base_vocab = load_base_checkpoint(args.from_checkpoint)
        if base_vocab != vocab_size:
            raise ValueError(
                f"Tokenizer vocab ({vocab_size}) != base checkpoint vocab "
                f"({base_vocab}); use the same tokenizer the base was trained with."
            )
        seq_len = min(args.seq_len, cfg.max_seq_len)
    else:
        if master:
            logger.warning(
                "No --from-checkpoint given: initializing from scratch on code "
                "(this is training-on-code, not fine-tuning)."
            )
        cfg = configure_vocab_size(mythos_1b())  # sizes vocab from the tokenizer
        cfg.max_seq_len = args.seq_len
        seq_len = args.seq_len
        model = OpenMythos(cfg)

    # Gradient checkpointing (opt-in). It recomputes each recurrent-loop
    # iteration in backward to fit longer sequences in memory, but the backward
    # recompute only routes deterministically if the MoE router bias does not
    # change between the forward and its recompute — so enabling it disables the
    # load-balancing bias update for this run.
    cfg.grad_checkpoint = args.grad_checkpoint
    if args.grad_checkpoint:
        from open_mythos.main import MoEFFN

        disabled = 0
        for mod in model.modules():
            if isinstance(mod, MoEFFN) and mod.bias_update_speed > 0:
                mod.bias_update_speed = 0.0
                disabled += 1
        if disabled and master:
            logger.warning(
                "grad checkpointing enabled -> MoE load-balancing bias update "
                "disabled for this run (the two are currently incompatible)."
            )

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    if ddp:
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype
            ),
            auto_wrap_policy=ModuleWrapPolicy({TransformerBlock, RecurrentBlock}),
            device_id=local_rank,
        )
        amp_ctx = nullcontext()
    else:
        model = model.to(device)
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if "cuda" in device
            else nullcontext()
        )

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Params: {n_params:,} | vocab {vocab_size:,} | seq_len {seq_len} | "
            f"grad_checkpoint={cfg.grad_checkpoint} | amp {amp_dtype}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )

    # Resume an interrupted fine-tuning run from --out-dir (model + optimizer).
    start_step = 0
    existing = _list_ckpts(args.out_dir)
    if existing:
        if master:
            logger.info(f"Resuming fine-tune from: {existing[-1]}")
        start_step = load_checkpoint(model, optimizer, existing[-1], ddp)

    dataset = PackedCodeDataset(
        encoding,
        seq_len,
        args.dataset,
        args.dataset_config,
        args.dataset_split,
        args.text_field,
        rank,
        world_size,
    )
    loader = DataLoader(
        dataset, batch_size=args.micro_batch, num_workers=4, pin_memory=True
    )

    def batch_iter():
        while True:
            yield from loader

    global_batch_tok = world_size * args.micro_batch * args.grad_accum * seq_len
    train_loop(
        model,
        optimizer,
        batch_iter(),
        cfg=cfg,
        vocab_size=vocab_size,
        device=device,
        total_steps=args.steps,
        start_step=start_step,
        grad_accum=args.grad_accum,
        warmup_steps=args.warmup,
        max_lr=args.lr,
        global_batch_tok=global_batch_tok,
        amp_ctx=amp_ctx,
        ddp=ddp,
        local_rank=local_rank,
        master=master,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        ckpt_dir=args.out_dir,
    )

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if master:
        logger.success("Fine-tuning complete.")


if __name__ == "__main__":
    main()
