# vllm server
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve verl-team/GenRM-CI-Test-1.5B --served_model_name genrm-demo

# sglang server
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang_router.launch_server --model-path verl-team/GenRM-CI-Test-1.5B --dp-size 4

set -x
# set max sub batch size for llm reward
export RULER_MAX_SUB_BATCH=16
export RULER_CONCURRENT_CHUNKS=32
export RULER_MAX_COMPLETION_TOKENS=10240

tool_config_path=examples/sglang_multiturn/config/tool_config/ugreen_mcp_tool_config.yaml

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${HOME}/data/ugreen_function_call/train.parquet \
    data.val_files=${HOME}/data/ugreen_function_call/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.tool_config_path=${tool_config_path} \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=2.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=./verl/utils/reward_score/llm_reward/llm_score.py \
    custom_reward_function.name=compute_score_batch \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir=rollout_data \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_func_rm_example_qwen3_1_7b' \
    trainer.experiment_name='qwen3_1.7b_function_gen_rm_1230' \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    trainer.resume_mode='disable'
