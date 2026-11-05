#!/usr/bin/env bash
# GRPO | Mistral Small 4 (119B-A6B) text | vLLM rollout | FSDP2 training
# Run from the repository root after preparing GSM8K and MATH parquet files.
# Source: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
# Load the official FP8 checkpoint directly: FSDP dequantizes it for training,
# while vLLM keeps native static FP8. Weight sync recomputes weight scales and
# retains the checkpoint-calibrated activation scales. No offline conversion.
# Requires Transformers with mistral4 support and vLLM >= 0.20.
# Defaults are starting values for 8 GPUs; memory requirements need validation.

set -xeuo pipefail

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-mistralai/Mistral-Small-4-119B-2603}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-4096}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
reasoning_effort=${REASONING_EFFORT:-none}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-4}
# Bound KV-cache planning to the training context, rather than the model's 1M default.
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_grpo_gsm8k_math}
experiment_name=${EXPERIMENT_NAME:-mistral_small_4_119b_vllm_fsdp2}
# ---- end user-adjustable ----

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
math_train=$HOME/data/math/train.parquet
math_test=$HOME/data/math/test.parquet

train_files="['$gsm8k_train', '$math_train']"
val_files="['$gsm8k_test', '$math_test']"
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    +data.apply_chat_template_kwargs.reasoning_effort=${reasoning_effort}
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    # Use the standard padded forward until mistral4 packing is validated.
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=0.7
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console", "swanlab"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
)

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): on GPU, the driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the vllm × fsdp extras of the committed uv.lock;
# NPU falls back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
