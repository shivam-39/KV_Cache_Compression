# Activate the virtual environment
# source /home/cse240d-fal26-lora/GEAR/cse240venv/bin/activate

# python evaluation_gsm8k_true_compression.py \
#   --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#   --prompt_file gsm8k_prompt_original.txt \
#   # --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 8 \
#   --quantize_bit 4 \
#   --rank 1 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 4 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python evaluation_gsm8k_true_compression.py \
#   --model meta-llama/Meta-Llama-3-8B \
#   --prompt_file gsm8k_prompt_original.txt \
#   --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 4 \
#   --quantize_bit 4 \
#   --rank 0 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python evaluation_aqua_cot_true_compression.py \
#   --model meta-llama/Meta-Llama-3-8B \
#   # --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 4 \
#   --quantize_bit 2 \
#   --rank 16 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python evaluation_bbh_cot_true_compression.py \
#   --model meta-llama/Meta-Llama-3-8B \
#   # --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 8 \
#   --quantize_bit 4 \
#   --rank 8 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256




##############################################################################################################

  #coherent text generation test
#   cd GenerationBench/GenerationTest

# python long_text_generation.py \
#   --prompt_file prompts/my_question.txt \
#   --output_file outputs/my_generation.txt \
#   --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --quantize_bit 4 \
#   --rank 0 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 4 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python scrolls_test.py \
#   --scrolls_subset gov_report \
#   --example_subset 0:2 \
#   --max_new_tokens 512 \
#   --model_max_length 4096 \
#   --compress_method None



##############################################################################################################
# Mistral tests


# python evaluation_gsm8k_true_compression.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 \
#   --prompt_file gsm8k_prompt_original.txt \
#   --example_subset 0:2 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 4 \
#   --quantize_bit 4 \
#   --rank 0 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python evaluation_aqua_cot_true_compression.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 \
#   --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 4 \
#   --quantize_bit 4 \
#   --rank 0 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256


# python evaluation_bbh_cot_true_compression.py \
#   --model mistralai/Mistral-7B-Instruct-v0.3 \
#   --task web_of_lies \
#   # --example_subset 0:10 \
#   --compress_method GEAR \
#   --compress_mode gear \
#   --batch_size 8 \
#   --quantize_bit 2 \
#   --rank 8 \
#   --loop 3 \
#   --left 0.02 \
#   --sink_tokens 16 \
#   --recency_tokens 64 \
#   --buffer_len 20 \
#   --max_new_tokens 256



##############################################################################################################

python calculate_adaptive_rank_compression.py \
  --seq-len 256 \
  --quant-bits 2 \
  --sink-recency 16 \
  # --model-preset llama3-8b
  --model-preset mistral-7b
  