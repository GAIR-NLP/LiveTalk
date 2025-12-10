#!/bin/bash

echo "=========================================="
echo "starting LoRA merging..."
echo "=========================================="
echo ""

torchrun --standalone --nproc_per_node=1 scripts/merge_lora.py --config configs/merge_lora_1B.yaml