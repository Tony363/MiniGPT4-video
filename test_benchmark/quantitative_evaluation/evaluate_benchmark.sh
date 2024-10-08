#!/bin/bash
set -o allexport
source /home/tony/MiniGPT4-video/.env 
set +o allexport
echo $API_KEY
# Define common arguments for all scripts
<<<<<<< HEAD
PRED_GENERIC="/home/tony/MiniGPT4-video/gpt_evaluation/mistral_finetune_test_config_eval.json"
OUTPUT_DIR="/home/tony/MiniGPT4-video/gpt_evaluation/mistral_finetune_test_config_eval"

NUM_TASKS=14
=======
PRED_GENERIC="/home/tony/MiniGPT4-video/gpt_evaluation/mistral_base_test_config_eval.json"
OUTPUT_DIR="/home/tony/MiniGPT4-video/gpt_evaluation/mistral_base_test_config_eval"

NUM_TASKS=14


>>>>>>> 7ee24512cec60775ff9cc2794956508e6d119a57

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
python evaluate_benchmark_5_consistency.py \
  --pred_path "${PRED_GENERIC}" \
  --output_dir "${OUTPUT_DIR}/consistency_eval" \
  --output_json "${OUTPUT_DIR}/consistency_results.json" \
  --api_key $API_KEY \
  --num_tasks $NUM_TASKS


echo "All evaluations completed!"
