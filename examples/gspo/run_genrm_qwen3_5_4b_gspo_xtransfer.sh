!#/usr/bin/bash env

set -x

cd verl
git checkout agent-rl-for-qwen3.5
cp -rvf ../scripts/verl/examples/* examples/

export TORCH_DISTRIBUTED_BACKEND=nccl
export TORCH_CUDA_ARCH_LIST="8.0"
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export OPENAI_BASE_URL="https://api9.xhub.chat/v1"
export OPENAI_API_BASE="https://api9.xhub.chat/v1"
export ANTHROPIC_BASE_URL="https://api9.xhub.chat/"
export ANTHROPIC_API_BASE="https://api9.xhub.chat/"
export GEMINI_BASE_URL="https://api9.xhub.chat/v1"
export GEMINI_API_BASE="https://api9.xhub.chat/v1"
export no_proxy=127.0.0.1,localhost,10.208.30.116
export http_proxy=http://nlb-w9tsm0845bm9hjgtpz.cn-shanghai.nlb.aliyuncsslb.com:1883
export https_proxy=http://nlb-w9tsm0845bm9hjgtpz.cn-shanghai.nlb.aliyuncsslb.com:1883


# vllm server
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve verl-team/GenRM-CI-Test-1.5B --served_model_name genrm-demo

# sglang server
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang_router.launch_server --model-path verl-team/GenRM-CI-Test-1.5B --dp-size 4

export RULER_JUDGE_MODEL="gpt-5.4-mini"
export RULER_MAX_SUB_BATCH=12
export RULER_CONCURRENT_CHUNKS=16
export RULER_MAX_COMPLETION_TOKENS=20480
export WANDB_API_KEY=${TRAIN_WANDB_API_KEY}
export OPENAI_API_KEY=${TRAIN_OPENAI_API_KEY}
model="/data/share/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
wandb login "${WANDB_API_KEY:?WANDB_API_KEY is required}"

tool_config_path=examples/sglang_multiturn/config/tool_config/ugreen_mcp_tool_config.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data/share/ml/data/ugreen_function_call/train.parquet \
    data.val_files=/data/share/ml/data/ugreen_function_call/test.parquet \
    data.train_max_samples=1024 \
    data.return_raw_chat=False \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.truncation='error' \
    data.tool_config_path=${tool_config_path} \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop="tool_agent" \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.actor.strategy=fsdp2 \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']"\
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1.05 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${tool_config_path} \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning-parser="qwen3" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.load_format="auto" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser="qwen3_coder" \
    actor_rollout_ref.ref.strategy=fsdp2 \
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']"\
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=./verl/utils/reward_score/llm_reward/llm_score.py \
    custom_reward_function.name=compute_score_batch \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir=rollout_data_qwen3_5_4b \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_func_rm_example_qwen3_5_4b' \
    trainer.experiment_name='qwen3_5_4b_function_gen_rm_test_gspo' \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.resume_mode='auto'

