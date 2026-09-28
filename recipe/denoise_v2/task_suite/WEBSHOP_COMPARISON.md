# WebShop 数据对齐与 W&B 核对（2026-09-28）

本次仅对齐 GiGPO 公开代码的数据与任务划分，不宣称完整复现论文训练协议。参照固定提交 [`f974dc5`](https://github.com/langfengQ/verl-agent/tree/f974dc5977908d6007ad178c9f3c48f0c7b39331)，不随 upstream 默认分支变化。

## 数据与任务

- 默认 profile：`gigpo_small`。商品 `items_shuffle_1000.json`、属性 `items_ins_v2_1000.json`，`human_goals=False`。对应[官方配置](https://github.com/langfengQ/verl-agent/blob/f974dc5977908d6007ad178c9f3c48f0c7b39331/verl/trainer/config/ppo_trainer.yaml)的 `use_small=True`。
- 固定镜像的 1000 个商品经原生 `load_products` / `get_synthetic_goals` 函数生成 6910 个 goals。选项组合可对应不同 goals，因此商品数不等于任务数。此规模核对在 CPU 上运行原生函数，不包含 Lucene 或 GPU 训练验证。
- catalog seed=42；train `[500,6910)` 共 6410，test `[0,500)` 共 500，无 dev。按 goal ID 分区，不是按商品 ASIN 分区；与[官方环境代码](https://github.com/langfengQ/verl-agent/blob/f974dc5977908d6007ad178c9f3c48f0c7b39331/agent_system/environments/env_package/webshop/envs.py)一致。
- 独立索引 `search_engine/indexes_gigpo_small`，显式传给原生模拟器，避免小商品环境误用全量搜索索引。索引命名是本项目的隔离措施；官方用默认 `indexes`，其实际内容取决于先前的数据准备步骤。
- 新任务目录 `recipe/denoise_v2/local_data/webshop_gigpo_small`。新默认实验名带 `_gigpo_small`，启动时校验 manifest profile，不能直接恢复旧任务池 checkpoint。
- 旧清单无 `data_profile` 时仍解释为 `full_human`；复评旧 checkpoint 要显式选择该 profile 和旧数据目录。

## 当前 W&B 结果

读取 [GRPO baseline run](https://wandb.ai/1005389104-beihang-university/denoise_v2_webshop/runs/vqf4zds9-0) 的 history，快照覆盖训练 step 0–248。`env.denoise.enable=False`，`algorithm.adv_estimator=grpo`，所以这是 GRPO baseline。

| 训练 step | test 成功率 | 原生平均 Score ×100 |
| --- | ---: | ---: |
| 0 | 4.4% | 12.53 |
| 25 | 29.0% | 55.92 |
| 50 | 31.0% | 58.94 |
| 75 | 32.8% | 54.90 |
| 100 | 38.0% | 62.62 |
| 125 | 38.8% | 66.51 |
| 150 | 40.4% | 66.48 |
| 175 | 39.6% | 63.78 |
| 200 | 39.2% | 63.46 |
| 225 | **40.8%** | **64.99** |

快照中最高成功率为 step 225 的 40.8%（204/500）；最高平均 score 为 step 125 的 66.51，不能把两个不同 checkpoint 的最优指标拼成同一个模型结果。每次 test 报告 500 个唯一任务、覆盖率 1。

训练指标未见明显崩溃：所有已记录 `episode/reward/mean` 等于 `10 × episode/success_rate`，数值型 history 未发现 NaN/Inf。step 176–225 平均动作**格式**有效率约 99.88%，response 长度截断率约 0.082%；达到环境步数上限的比例约 10.59%。仍存在任务失败和超时，不能将格式有效率当成功率。

现有 run 的 TaskSuite 后端使用全量商品和 human goals，旧训练分区从 1500 开始。W&B 中遗留的 `env.webshop.use_small=True/human_goals=False` 是另一套环境入口的配置字段，当时并不控制 TaskSuite；不能据此认定旧 run 已采用小商品集。本次默认启动显式记录并校验 `env.task_suite.webshop_data_profile`。

旧 [DenoiseRL run](https://wandb.ai/1005389104-beihang-university/denoise_v2_webshop/runs/bwlgef0l-0) 的最近 dev 成功率为 43.4%、score 为 71.05（step 225，1000 个 dev goals）。它采用不同 split、旧奖励及 greedy 评测，不能直接与上表比较算法优劣。

## 如何与论文对照

同为 Qwen2.5-7B-Instruct，[论文 Table 1](https://arxiv.org/html/2505.10978v1#S5.T1) 中 GRPO 的 success 为 66.1±3.7%、score 为 79.3±2.8；GiGPO w/o std 的 success 为 75.2±3.8%、score 为 86.2±2.6，均汇总 3 个种子。当前数值更低，但不同数据与评测协议下，不能把差值解释成算法或实现损失；也不能保证切换小商品集后就达到论文数字。

准确率应汇报 **`val/test/success_rate × 100%`**；另报 **`val/test/test_score × 100`** 作为平均 Score。训练 reward=10×训练成功率，不能当作测试准确率。按 test 选择最佳 checkpoint 时，应注明选择方式。

按本次要求保留的差异如下。官方值来自[对应 GRPO 示例](https://github.com/langfengQ/verl-agent/blob/f974dc5977908d6007ad178c9f3c48f0c7b39331/examples/grpo_trainer/run_webshop.sh)，并非此次要修改的目标：

| 设置 | 本项目保持 | 官方示例 |
| --- | --- | --- |
| solver | 7B | 示例默认 1.5B；论文另有 7B |
| 每题 rollout | 16 | 8 |
| 每步 response 上限 | 256 | 512 |
| PPO mini-batch | 128 | 64 |
| 训练更新预算 | 500 | 150 |
| 评测 | 完整 500 个 test goals | 每次从 500 个池中抽取 128 个 |
| 评测 temperature / top_p | 0.6 / 0.95 | 0.4 / 1.0 |
| 训练任务调度 | 保留任务池遍历 | batch 内不重复，跨 batch 可重复抽取 |

最大步数均为 15，历史长度均为 2，训练奖励均为成功 10/失败 0。此次没有改上述保留项、rho 分组/控制或 latest+best checkpoint 规则。

## 在训练服务器应用

先同步本次代码，再在已有 `denoise-webshop` 环境中准备独立数据目录：

```bash
cd /inspire/hdd/global_user/xucaijun-253108120121/DenoiseRL-agent
source /inspire/hdd/global_user/xucaijun-253108120121/miniconda/bin/activate denoise-webshop
python -m recipe.denoise_v2.task_suite.prepare_webshop_assets --download --build-index
python -m recipe.denoise_v2.task_suite.prepare_tasks \
  --benchmark webshop --train-batch-size 16 \
  --output recipe/denoise_v2/local_data/webshop_gigpo_small
python -m recipe.denoise_v2.task_suite.smoke_env \
  --benchmark webshop --manifest recipe/denoise_v2/local_data/webshop_gigpo_small/tasks.json
```

新开 GRPO baseline：

```bash
WANDB_MODE=online EXPERIMENT_NAME=webshop_grpo_7b_seed0_gigpo_small \
  bash recipe/denoise_v2/run_webshop_grpo_train.sh --seed 0 \
  --data-dir recipe/denoise_v2/local_data/webshop_gigpo_small
```

DenoiseRL 使用相同数据目录，换为 `run_webshop_denoise_train.sh` 和独立实验名。已运行的旧作业不会因本地代码修改而自动切换数据。本次没有重启服务器作业，也没有实际执行 GPU 训练；服务器上先完成上面的原生环境 smoke test。
