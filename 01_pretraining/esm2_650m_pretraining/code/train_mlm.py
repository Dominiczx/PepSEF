#!/usr/bin/env python3
"""
train_mlm.py

Trainer script to continue pretraining / fine-tuning ESM-2 (masked language modeling)
Supports two-stage workflow:
  stage1 : train on full protein sequences (protein_merge_80.fasta)
  stage2 : train on combination of peptides and sliding-window protein slices

This script is written to be launched with `accelerate launch` (recommended) or run
single-process for debugging.

Example accelerate launch (8 GPUs):
  accelerate launch --num_processes 8 train_mlm.py --stage 1 --input /path/to/protein_merge_80.fasta \
      --model facebook/esm2_t33_650M_UR50D --output_dir outputs/stage1 --epochs 3 --per_device_batch_size 16 --lr 5e-5

Notes:
 - Uses the pretrained ESM-2 tokenizer from Hugging Face hub via AutoTokenizer.
 - Uses Hugging Face Trainer + DataCollatorForLanguageModeling for MLM.
"""
import argparse
import os
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
import random
import math
import torch.distributed as dist
# Optional PEFT/LoRA imports (used only if --use_lora)
try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    # PEFT may not be installed in the environment; the script will raise a helpful error when --use_lora is enabled
    LoraConfig = None
    get_peft_model = None
    TaskType = None


def read_fasta_sequences(path):
    sequences = []
    header = None
    seq_lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences.append(''.join(seq_lines))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            sequences.append(''.join(seq_lines))
    return sequences


def make_dataset_from_fasta(path):
    seqs = read_fasta_sequences(path)
    # filter short sequences
    seqs = [s for s in seqs if len(s) > 0]
    return Dataset.from_dict({"sequence": seqs})


def make_stage2_dataset(peptides_path, proteins_path, min_len=20, max_len=50, seed=42, save_to=None):
    """Create a stage-2 dataset by taking all peptides and randomly sampling+cropping
    proteins to produce a 1:1 match of protein crops to peptides.

    This version samples crop lengths biased towards shorter lengths (peptide-like)
    using a triangular distribution with mode at min_len, and uses a random stride
    when selecting a start position to introduce variability and bias toward peptide-like
    fragments.
    Returns a Hugging Face Dataset with key 'sequence'.
    """
    peptides = read_fasta_sequences(peptides_path)
    proteins = read_fasta_sequences(proteins_path)
    if len(peptides) == 0:
        raise ValueError(f"No peptides found in {peptides_path}")
    if len(proteins) == 0:
        raise ValueError(f"No proteins found in {proteins_path}")

    random.seed(seed)
    protein_crops = []
    attempts = 0
    target = len(peptides)
    while len(protein_crops) < target:
        seq = random.choice(proteins)
        attempts += 1
        if len(seq) < min_len:
            if attempts > target * 10:
                # fallback: skip short proteins after many attempts
                continue
            else:
                continue
        # sample crop length biased toward min_len using triangular distribution
        max_len_eff = min(max_len, len(seq))
        L = int(random.triangular(min_len, max_len_eff, min_len))
        # choose a random stride (1..max_stride) and pick a start aligned to that stride
        max_stride = max(1, min(10, L // 2))
        stride = random.randint(1, max_stride)
        start_candidates = list(range(0, max(1, len(seq) - L + 1), stride))
        if not start_candidates:
            start = 0
        else:
            start = random.choice(start_candidates)
        crop = seq[start:start+L]
        protein_crops.append(crop)

    combined = list(peptides) + protein_crops
    random.shuffle(combined)

    if save_to is not None:
        # optionally write to fasta file
        with open(save_to, 'w') as f:
            for i, seq in enumerate(combined):
                f.write(f">seq_{i}\n")
                for j in range(0, len(seq), 80):
                    f.write(seq[j:j+80] + "\n")

    return Dataset.from_dict({"sequence": combined})


def tokenize_function(examples, tokenizer, max_length=None):
    # tokenizer for ESM expects sequences as raw text
    kwargs = dict(truncation=True)
    if max_length is not None:
        kwargs["max_length"] = max_length
    return tokenizer(examples["sequence"], **kwargs)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--input", type=str, required=False, default=None, help="Input FASTA for the selected stage (not required for stage 2 when --peptides and --proteins are provided)")
    parser.add_argument("--model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--output_dir", type=str, default="outputs/esm2_finetune")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--fp16", action='store_true', help='Use AMP mixed precision')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='Number of steps to accumulate gradients')

    # LoRA/PEFT arguments
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adapters for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj", help="Comma-separated target modules for LoRA (e.g., q_proj,v_proj)")
    parser.add_argument("--lora_save_steps", type=int, default=50, help="Save LoRA checkpoints every N steps")
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM", help="PEFT TaskType name (optional)")

    # Stage-2 dataset creation args (if using --stage 2 and providing peptides & proteins)
    parser.add_argument("--peptides", type=str, default=None, help="Peptide FASTA path (used for stage 2)")
    parser.add_argument("--proteins", type=str, default=None, help="Protein FASTA path (used for stage 2)")
    parser.add_argument("--protein_crop_min", type=int, default=20, help="Minimum crop length for proteins (stage2)")
    parser.add_argument("--protein_crop_max", type=int, default=50, help="Maximum crop length for proteins (stage2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--save_steps", type=int, default=50, help="Save model every N steps")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop training after this many steps (useful for quick tests)")
    parser.add_argument("--val_split", type=float, default=0.0, help="Fraction of data to hold out for validation (0 disables)")
    parser.add_argument("--val_steps", type=int, default=10, help="Run validation every N steps")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps for LR scheduler")
    parser.add_argument("--max_val_batches", type=int, default=0, help="Maximum number of validation batches to run (0 = all)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True)

    # Optionally wrap model with LoRA adapters via PEFT
    if args.use_lora:
        if get_peft_model is None or LoraConfig is None:
            raise ImportError("PEFT is required for LoRA. Install it with `pip install peft` and rerun with --use_lora")
        # parse requested target modules (comma-separated)
        requested = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

        # collect all module names from the base model
        module_names = [name for name, _ in model.named_modules()]

        # try to resolve requested names to actual module paths
        resolved = []
        for req in requested:
            # exact match
            if req in module_names:
                resolved.append(req)
                continue
            # match by suffix or containing token
            matches = [n for n in module_names if req in n or n.endswith(req) or n.split('.')[-1] == req]
            if matches:
                resolved.extend(matches)

        # fallback: if nothing resolved, try common attention q/v names used in ESM models
        if not resolved:
            resolved = [n for n in module_names if n.endswith('attention.self.query') or n.endswith('attention.self.value') or 'attention.self.q_proj' in n or 'attention.self.v_proj' in n]

        # dedupe and keep ordering
        seen = set()
        target_modules = [x for x in resolved if not (x in seen or seen.add(x))]

        if not target_modules:
            raise ValueError(f"Unable to resolve any target modules for LoRA from: {args.lora_target_modules}")

        # If TaskType is available, try to resolve string to enum
        task_type = None
        try:
            if TaskType is not None and hasattr(TaskType, args.lora_task_type):
                task_type = getattr(TaskType, args.lora_task_type)
        except Exception:
            task_type = None

        if task_type is not None:
            lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules, lora_dropout=args.lora_dropout, bias="none", task_type=task_type)
        else:
            lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules, lora_dropout=args.lora_dropout, bias="none")

        model = get_peft_model(model, lora_config)
        print(f"LoRA enabled (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}) on modules: {target_modules}")

    print("Preparing dataset...")
    if args.stage == 2 and args.peptides and args.proteins:
        print(f"Building stage-2 dataset from peptides={args.peptides} and proteins={args.proteins} (min={args.protein_crop_min}, max={args.protein_crop_max})")
        dataset = make_stage2_dataset(args.peptides, args.proteins, min_len=args.protein_crop_min, max_len=args.protein_crop_max, seed=args.seed, save_to=os.path.join(args.output_dir, 'stage2_combined.fasta'))
    else:
        dataset = make_dataset_from_fasta(args.input)

    # optional validation split (done before tokenization)
    val_dataset = None
    if args.val_split and args.val_split > 0.0:
        if args.val_split <= 0.0 or args.val_split >= 0.5:
            raise ValueError("--val_split should be >0.0 and <0.5")
        split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        train_ds = split['train']
        val_dataset = split['test']
        print(f"Using validation split: {len(val_dataset)} samples (fraction {args.val_split})")
    else:
        train_ds = dataset

    # tokenize
    tokenized_train = train_ds.map(lambda examples: tokenizer(examples["sequence"], truncation=True, padding=False, max_length=args.max_length), batched=True, remove_columns=["sequence"])
    tokenized_val = None
    if val_dataset is not None:
        tokenized_val = val_dataset.map(lambda examples: tokenizer(examples["sequence"], truncation=True, padding=False, max_length=args.max_length), batched=True, remove_columns=["sequence"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)

    # Distributed vs DataParallel handling
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1

    if distributed:
        # initialize process group (assumes torchrun/env launch)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{local_rank}')
        is_main = (local_rank == 0)
        print(f"Distributed training initialized (rank {local_rank}/{world_size})") if is_main else None
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        n_gpus = torch.cuda.device_count()
        use_data_parallel = n_gpus > 1
        is_main = True

    # DataLoader using the HuggingFace data collator
    if distributed:
        sampler = DistributedSampler(tokenized_train, num_replicas=world_size, rank=local_rank, shuffle=True)
        dataloader = DataLoader(tokenized_train, batch_size=args.per_device_batch_size, sampler=sampler, collate_fn=lambda x: data_collator(x))
    else:
        if use_data_parallel:
            effective_batch_size = args.per_device_batch_size * max(1, n_gpus)
            dataloader = DataLoader(tokenized_train, batch_size=effective_batch_size, shuffle=True, collate_fn=lambda x: data_collator(x))
        else:
            dataloader = DataLoader(tokenized_train, batch_size=args.per_device_batch_size, shuffle=True, collate_fn=lambda x: data_collator(x))

    val_dataloader = None
    if tokenized_val is not None:
        # choose val batch size to match global per-step batch so DataParallel uses all GPUs
        if distributed:
            val_batch_size = args.per_device_batch_size * world_size
        elif use_data_parallel:
            val_batch_size = effective_batch_size if 'effective_batch_size' in locals() else args.per_device_batch_size * max(1, n_gpus)
        else:
            val_batch_size = args.per_device_batch_size

        # reasonable number of workers for faster loading
        try:
            num_workers = max(0, min(4, (os.cpu_count() or 1) - 1))
        except Exception:
            num_workers = 0

        val_dataloader = DataLoader(tokenized_val, batch_size=val_batch_size, shuffle=False, collate_fn=lambda x: data_collator(x), num_workers=num_workers)

    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main:
            print(f"Using DDP across {world_size} processes")
    elif use_data_parallel:
        print(f"{n_gpus} GPUs detected — using torch.nn.DataParallel")
        model = torch.nn.DataParallel(model)
    else:
        print(f"Using device: {device}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(args.fp16 and torch.cuda.is_available()))
    # LR scheduler will be created after batch size / steps per epoch is known
    scheduler = None

    print("Starting training loop...")
    model.train()
    grad_accum = int(args.gradient_accumulation_steps)

    total_samples = len(tokenized_train)
    if distributed:
        # global batch per step is per_device_batch_size * world_size
        per_step_global_bs = args.per_device_batch_size * world_size
    else:
        per_step_global_bs = effective_batch_size if 'effective_batch_size' in locals() else args.per_device_batch_size

    # Now that per-step global batch is known, create scheduler if possible
    try:
        steps_per_epoch = math.ceil(len(tokenized_train) / per_step_global_bs)
    except Exception:
        steps_per_epoch = None
    if args.max_steps is not None:
        num_training_steps = int(args.max_steps)
    else:
        if steps_per_epoch is None:
            num_training_steps = None
        else:
            num_training_steps = int(steps_per_epoch * args.epochs)
    if num_training_steps is not None and num_training_steps > 0:
        try:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
        except Exception:
            scheduler = None

    if is_main:
        print(f"Stage {args.stage} — total samples: {total_samples} — global batch/step: {per_step_global_bs}")

    global_step = 0
    processed_samples = 0
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    metrics_path = os.path.join(args.output_dir, 'epoch_metrics.csv')
    if is_main and not os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'w') as mf:
                mf.write('epoch,avg_train_loss,avg_masked_acc,last_val_loss,global_step\n')
        except Exception as ex:
            print('Warning: failed to initialize epoch metrics file:', ex)

    for epoch in range(int(args.epochs)):
        if distributed:
            # set epoch for sampler shuffling
            dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_acc_sum = 0.0
        epoch_acc_steps = 0
        epoch_last_val_loss = None
        processed_in_accum = 0
        optimizer.zero_grad()
        for batch in dataloader:
            # collator returns tensors; move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            # infer current batch size (per process)
            try:
                batch_size_current = batch['input_ids'].size(0)
            except Exception:
                # fallback
                batch_size_current = next(iter(batch.values())).size(0)

            # forward pass
            if args.fp16 and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss_raw = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    # ensure loss_raw is a scalar (e.g., DataParallel may return per-device losses)
                    try:
                        if hasattr(loss_raw, 'dim') and loss_raw.dim() > 0:
                            loss_raw = loss_raw.mean()
                    except Exception:
                        pass
                    try:
                        loss_value = float(loss_raw.item())
                    except Exception:
                        loss_value = float(loss_raw.mean().item())
                    loss = loss_raw / grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss_raw = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                # ensure loss_raw is a scalar before backward (handles DataParallel returning per-device losses)
                try:
                    if hasattr(loss_raw, 'dim') and loss_raw.dim() > 0:
                        loss_raw = loss_raw.mean()
                except Exception:
                    pass
                try:
                    loss_value = float(loss_raw.item())
                except Exception:
                    loss_value = float(loss_raw.mean().item())
                loss = loss_raw / grad_accum
                loss.backward()

            processed_in_accum += batch_size_current

            # optimizer step and scaler update only after grad_accum steps
            if (epoch_steps + 1) % grad_accum == 0:
                if args.fp16 and torch.cuda.is_available():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    try:
                        scheduler.step()
                    except Exception:
                        pass

                # Update global step and processed samples (global view)
                global_step += 1
                processed_samples += per_step_global_bs
                processed_in_accum = 0

                # compute accuracy on masked positions
                with torch.no_grad():
                    logits = outputs.logits
                    labels = batch.get('labels')
                    accuracy = 0.0
                    if labels is not None:
                        mask = labels != -100
                        if mask.sum().item() > 0:
                            preds = logits.argmax(dim=-1)
                            correct = ((preds == labels) & mask).sum().item()
                            total_masked = mask.sum().item()
                            accuracy = correct / total_masked
                            epoch_acc_sum += float(accuracy)
                            epoch_acc_steps += 1

                # Run validation if requested
                current_val_loss = None
                if is_main and val_dataloader is not None and args.val_steps > 0 and (global_step % args.val_steps == 0):
                    model.eval()
                    val_loss_acc = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        vb_count = 0
                        for vbatch in val_dataloader:
                            vbatch = {k: v.to(device) for k, v in vbatch.items()}
                            vout = model(**vbatch)
                            vloss = vout.loss if hasattr(vout, 'loss') else vout[0]
                            try:
                                if hasattr(vloss, 'dim') and vloss.dim() > 0:
                                    vloss = vloss.mean()
                            except Exception:
                                pass
                            try:
                                vval = float(vloss.item())
                            except Exception:
                                continue
                            if math.isnan(vval):
                                continue
                            val_loss_acc += vval
                            val_batches += 1
                            vb_count += 1
                            # limit validation batches to speed up validation when requested
                            if args.max_val_batches and args.max_val_batches > 0 and vb_count >= args.max_val_batches:
                                break
                    if val_batches > 0:
                        val_loss = val_loss_acc / val_batches
                        current_val_loss = val_loss
                        epoch_last_val_loss = val_loss
                        print(f"Validation — Step {global_step} — val_loss {val_loss:.4f}")
                    else:
                        print(f"Validation — Step {global_step} — val_loss (no valid batches)")
                    # switch back to train
                    model.train()

                percent = (processed_samples / total_samples * 100) if total_samples > 0 else 0.0
                if is_main:
                    print(f"[Stage {args.stage}] Epoch {epoch+1} Step {global_step} — processed {processed_samples}/{total_samples} ({percent:.2f}%) — loss {loss_value:.4f} — acc {accuracy:.4f}")

                    # Save LoRA adapters checkpoints periodically
                    if args.use_lora and (args.lora_save_steps > 0) and (global_step % args.lora_save_steps == 0):
                        save_dir = os.path.join(args.output_dir, f"lora_step_{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        try:
                            if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
                                model.module.save_pretrained(save_dir)
                            else:
                                model.save_pretrained(save_dir)
                            # measure size
                            size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(save_dir) for filename in filenames) / (1024.0 ** 2)
                            print(f"Saved LoRA checkpoint to {save_dir} (size: {size_mb:.2f} MB)")
                            if os.path.exists(args.model):
                                base_size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(args.model) for filename in filenames) / (1024.0 ** 2)
                                print(f"Base model directory size: {base_size_mb:.2f} MB")
                        except Exception:
                            print(f"Warning: failed to save LoRA checkpoint at step {global_step}")

                    # Save full model every save_steps (if not using LoRA, or to save full state)
                    if args.save_steps > 0 and (global_step % args.save_steps == 0):
                        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        try:
                            if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
                                # when using PEFT, model.module.save_pretrained saves adapters; if user wants full model, they'd need separate logic
                                model.module.save_pretrained(ckpt_dir)
                                tokenizer.save_pretrained(ckpt_dir)
                            else:
                                model.save_pretrained(ckpt_dir)
                                tokenizer.save_pretrained(ckpt_dir)
                            # measure size
                            size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(ckpt_dir) for filename in filenames) / (1024.0 ** 2)
                            print(f"Saved model checkpoint to {ckpt_dir} (size: {size_mb:.2f} MB)")
                        except Exception:
                            print(f"Warning: failed to save checkpoint at step {global_step}")

                    # Save best model based on validation loss when available, else training loss
                    improved = False
                    metric_name = 'train_loss'
                    metric_value = loss_value
                    if current_val_loss is not None:
                        metric_name = 'val_loss'
                        metric_value = current_val_loss
                        if metric_value < best_val_loss:
                            best_val_loss = metric_value
                            improved = True
                    else:
                        if metric_value < best_train_loss:
                            best_train_loss = metric_value
                            improved = True

                    if improved:
                        best_dir = os.path.join(args.output_dir, 'best')
                        os.makedirs(best_dir, exist_ok=True)
                        try:
                            if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
                                model.module.save_pretrained(best_dir)
                                tokenizer.save_pretrained(best_dir)
                            else:
                                model.save_pretrained(best_dir)
                                tokenizer.save_pretrained(best_dir)
                            size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(best_dir) for filename in filenames) / (1024.0 ** 2)
                            print(f"New best model saved to {best_dir} ({metric_name} {metric_value:.4f}, size: {size_mb:.2f} MB)")
                            if args.use_lora and os.path.exists(args.model):
                                base_size_mb = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, _, filenames in os.walk(args.model) for filename in filenames) / (1024.0 ** 2)
                                print(f"Base model directory size: {base_size_mb:.2f} MB")
                        except Exception:
                            print("Warning: failed to save best model")

                # optional: stop after a maximum number of steps
                if args.max_steps is not None and global_step >= args.max_steps:
                    if is_main:
                        print(f"Reached max_steps={args.max_steps}; stopping training")
                    stop_training = True
                    break
            epoch_loss += loss_value
            epoch_steps += 1

            if 'stop_training' in locals() and stop_training:
                break

        avg_loss = epoch_loss / max(1, epoch_steps)
        avg_acc = epoch_acc_sum / max(1, epoch_acc_steps)
        if is_main:
            print(f"Epoch {epoch+1}/{args.epochs} — avg loss: {avg_loss:.4f}")
            try:
                val_str = '' if epoch_last_val_loss is None else f'{epoch_last_val_loss:.6f}'
                with open(metrics_path, 'a') as mf:
                    mf.write(f"{epoch+1},{avg_loss:.6f},{avg_acc:.6f},{val_str},{global_step}\n")
            except Exception as ex:
                print('Warning: failed to append epoch metrics:', ex)

        if 'stop_training' in locals() and stop_training:
            break

        # Save a checkpoint for this epoch
        epoch_ckpt = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(epoch_ckpt, exist_ok=True)
        if isinstance(model, torch.nn.DataParallel):
            save_model = model.module
        else:
            save_model = model
        save_model.to('cpu')
        save_model.save_pretrained(epoch_ckpt)
        tokenizer.save_pretrained(epoch_ckpt)
        print(f"Saved checkpoint for epoch {epoch+1} to {epoch_ckpt}")
        save_model.to(device)

    # Save model (unwrapping DataParallel if used)
    if isinstance(model, torch.nn.DataParallel):
        save_model = model.module
    else:
        save_model = model
    save_model.to('cpu')
    print('Saving model to', args.output_dir)
    save_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # If LoRA was used, save the final adapters separately
    if args.use_lora:
        lora_dir = os.path.join(args.output_dir, 'lora_final')
        os.makedirs(lora_dir, exist_ok=True)
        # save_pretrained should include only the adapter weights/config
        try:
            save_model.save_pretrained(lora_dir)
            print(f"Saved final LoRA adapters to {lora_dir}")
        except Exception:
            # fallback: if model was wrapped in DataParallel earlier, try module
            if isinstance(model, torch.nn.DataParallel):
                model.module.save_pretrained(lora_dir)
                print(f"Saved final LoRA adapters to {lora_dir} (via model.module)")

    print("Finished training. Model and tokenizer saved to:", args.output_dir)


if __name__ == '__main__':
    main()
