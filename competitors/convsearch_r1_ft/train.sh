#!/bin/bash
# GRPO stage of ConvSearch-R1 on the ConvRevise training conversations: the authors' verl recipe (examples/grpo_trainer/
# run_qwen2.5-3b-it_topiocqa.sh of github.com/BeastyZ/ConvSearch-R1, commit e7854c89), started from their released checkpoint, so their SFT
# stage is not repeated, with the reward of competitors/convsearch_r1_ft/reward_server.py. One node with 4 GPUs, in the authors' verl
# environment (their README); the reward server runs in the retriever's environment on 4 other GPUs.
#
#   1. python -m competitors.convsearch_r1_ft.build_data                              -> outputs/convsearch_r1_ft/grpo/{train,val}.parquet
#   2. python -m competitors.convsearch_r1_ft.reward_server --port 8000               (retriever environment, another node)
#   3. CONVSEARCH_R1_REPO=<clone of github.com/BeastyZ/ConvSearch-R1> RETRIEVER_URL=http://<server host>:8000/retrieve \
#        bash competitors/convsearch_r1_ft/train.sh                                   -> models/convsearch_r1_ft_ckpt/global_step_39/
#   4. CONVSEARCH_R1_REPO=... bash competitors/convsearch_r1_ft/merge.sh               -> models/convsearch_r1_ft/
#
# Their settings are kept (GRPO, lr 1e-6, batch 128 = mini batch, micro batch 8, KL loss 0.001 low_var_kl, 16 rollouts at temperature 0.7,
# 1024 response tokens, gradient checkpointing, remove padding, vLLM rollout at 0.6 GPU memory, the rewrite_r1 reward manager, topk 100)
# except: max_prompt_length 2048 (theirs 1536: 36 of the 1,790 prompts are longer and are dropped by filter_overlong_prompts, which
# leaves 1,754 prompts = 13 steps of 128 per epoch); 10 warm-up steps (theirs 100); 4 GPUs instead of 8 (same batch); 39 steps, i.e. 3 of
# their 6 epochs, where the paper's run was stopped (the learning rate is constant after warm-up, so the cap changes no earlier step).
# A checkpoint is written after every step; running the script again resumes from the last one (resume_mode auto).
# verl reads the reward server's address from RETRIEVER_URL, not from retriever.url.
set -eu
ROOT=${MEMREVISE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
REPO=${CONVSEARCH_R1_REPO:?point CONVSEARCH_R1_REPO at a clone of github.com/BeastyZ/ConvSearch-R1}
export RETRIEVER_URL=${RETRIEVER_URL:?the reward server, http://<host>:<port>/retrieve}
export VLLM_ATTENTION_BACKEND=XFORMERS
DATA=$ROOT/outputs/convsearch_r1_ft/grpo
cd "$REPO/verl"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA/train.parquet \
    data.val_files=$DATA/val.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=BeastyZ/Qwen2.5-3B-ConvSearch-R1-TopiOCQA \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='convsearch_r1_ft' \
    trainer.experiment_name='convsearch_r1_ft' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=auto \
    trainer.remove_previous_ckpt_in_save=True \
    trainer.default_local_dir=$ROOT/models/convsearch_r1_ft_ckpt \
    trainer.total_epochs=6 \
    trainer.total_training_steps=39 \
    reward_model.reward_manager=rewrite_r1 \
    custom_reward_function.path=$REPO/verl/verl/utils/reward_score/rewrite_r1.py \
    retriever.url=$RETRIEVER_URL \
    retriever.topk=100
