CUDA_VISIBLE_DEVICES=0,1 python -u /home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/code/train_mlm.py \
  --stage 2 \
  --peptides /home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/data/peptides_merge.fasta \
  --proteins /home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/data/protein_merge_80.fasta \
  --model /home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/models/esm2-650M \
  --output_dir /home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/outputs/finetuned_stage2_full_external \
  --epochs 1000 --per_device_batch_size 8 --gradient_accumulation_steps 2 \
  --use_lora --lora_r 16 --lora_dropout 0.1 --lora_task_type FEATURE_EXTRACTION \
  --lora_target_modules q_proj,v_proj \
  --lr 3e-6 --warmup_steps 100 --val_split 0.05 --val_steps 50 --max_val_batches 16 --mlm_prob 0.15
  