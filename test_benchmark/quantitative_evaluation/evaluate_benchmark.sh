#!/bin/bash
set -o allexport
source /home/tony/MiniGPT4-video/openai_api.env 
set +o allexport
echo $API_KEY
# Define common arguments for all scripts
PRED_GENERIC="/home/tony/MiniGPT4-video/gpt_evaluation/llama2_test_config_eval.json"
OUTPUT_DIR="/home/tony/MiniGPT4-video/gpt_evaluation/llama2_test_config_eval"

# PRED_GENERIC="/home/tony/MiniGPT4-video/gpt_evalaution/mistral_test_config_eval.json"
# OUTPUT_DIR="/home/tony/MiniGPT4-video/gpt_evaluation"
# PRED_GENERIC="/home/tony/MiniGPT4-video/gpt_evaluation/llama2_test_config_eval.json"
# OUTPUT_DIR="/home/tony/MiniGPT4-video/gpt_evaluation"
# API_KEY=""
NUM_TASKS=14



# Run the "correctness" evaluation script
python evaluate_benchmark_1_correctness.py \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/correctness_eval" \
  --output_json "${OUTPUT_DIR}/correctness_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS


# Run the "detailed orientation" evaluation script
python evaluate_benchmark_2_detailed_orientation.py \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/detailed_eval" \
  --output_json "${OUTPUT_DIR}/detailed_orientation_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "contextual understanding" evaluation script
python evaluate_benchmark_3_context.py \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/context_eval" \
  --output_json "${OUTPUT_DIR}/contextual_understanding_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "temporal understanding" evaluation script
python evaluate_benchmark_4_temporal.py \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/temporal_eval" \
  --output_json "${OUTPUT_DIR}/temporal_understanding_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS

# Run the "consistency" evaluation script
# python evaluate_benchmark_5_consistency.py \
#   --pred_path "${PRED_GENERIC}" \
#   --output_dir "${OUTPUT_DIR}/consistency_eval" \
#   --output_json "${OUTPUT_DIR}/consistency_results.json" \
#   --api_key $API_KEY \
#   --num_tasks $NUM_TASKS


echo "All evaluations completed!"
