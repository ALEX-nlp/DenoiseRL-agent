# WebShop / ScienceWorld：GRPO baseline 与 DenoiseRL v2

这是 ALFWorld v2 的两个新环境实现。`sciworld` 在 Python 启动器中是 `scienceworld` 的别名。推荐配置面向 **8×80GB GPU，Qwen2.5-7B-Instruct solver，Qwen2.5-1.5B-Instruct 固定弱模型**，每批任务数与 rho 控制参数对齐 ALFWorld v2，未经过这两个 benchmark 的完整训练调优。

## 方法与实验口径

- baseline 是 **GRPO**：每个任务 16 条从初始状态开始的轨迹，不加载弱模型。
- DenoiseRL v2：每个任务至多生成一条弱模型轨迹，16 条 solver 续接共用截断前缀；`rho=0` 跳过弱模型生成。
- 两者使用相同的无放回任务遍历、rollout 数、solver、reward、prompt、训练步数和评估协议。baseline 的 `rho` 固定为 0，但仍保存任务遍历状态。
- WebShop 用具体 **goal index** 定位任务，默认按购买选项数与属性要求数划分六组共享 rho，也可选择按商品大类分组；ScienceWorld 用 **task name::variation ID** 定位任务，按 task name 共享 rho。
- WebShop 固定 catalog seed=42，所有 worker、train/dev/test 使用同一目标目录；训练随机种子只影响采样顺序。改变 catalog seed 会改变任务定义/顺序和数据划分，不能当作普通训练随机种子。
- `rho <- clip(rho + alpha * (本批该类型完全成功率 - target), 0, max_rho)`。先对 16 条轨迹求成功率，再对同类型任务等权平均；不使用部分得分更新 rho。
- 在线模式沿用 ALFWorld v2：**没有筛掉弱模型的成功轨迹**，所以前缀可能包含错误，也可能包含有效进展。
- 前缀动作重放进环境和历史；只有 solver 新动作参与 PPO。前缀消耗环境步数预算，遇到终止状态的前缀会报错；相同 task/prefix 返回不同状态也会报错。
- 两个方法都使用一次性的 **terminal score reward**：WebShop 为原生 0–1 得分，ScienceWorld 为终止/步数上限时的 `clip(score, 0, 100)/100`。中间步骤 reward=0，避免重复累加进度。原始 score=-1 的失败映射为 0。该定义会奖励继承前缀后最终达到的部分进度。
- 完全成功独立统计：WebShop score=1，ScienceWorld score=100。`env.task_suite.reward_mode=success` 可改为纯二值 reward；做比较时两组都要改。
- WebShop 新划分为 train `[1500, n)`、dev `[500,1500)`、test `[0,500)`，与 bundled upstream baseline 一致；不同于旧环境 wrapper 的 train `[500,n)`。ScienceWorld 直接使用官方 `get_variations_train/dev/test()`，不随机重划。
- 训练中只评估 dev。最终显式选择 test，全量遍历一次，断言实际 task ID、数量、重复次数与请求一致。评估从干净初始状态开始，不需要弱模型或 curriculum 文件。

## rho 分组的含义

| 环境 | 具体任务 ID | 共享 rho 的分组键 |
|---|---|---|
| ALFWorld | gamefile | `task_type`，如加热后放置、清洗后放置 |
| WebShop | goal index | 默认 `structure` 六组；可选 `category` 商品大类 |
| ScienceWorld | `task_name::variation` | `task_name`；同名任务的所有 variation 共享 rho |

ScienceWorld 的分组对应任务行为类型。WebShop 默认使用 `--webshop-rho-grouping structure`，按训练 goal 中的 `len(goal_options)` 与 `len(attributes)` 分组：

| 指定购买选项数 | 属性要求 1–2 个 | 属性要求 ≥3 个 |
|---|---|---|
| 0 个 | `options_0__attrs_1_2` | `options_0__attrs_3_plus` |
| 1 个 | `options_1__attrs_1_2` | `options_1__attrs_3_plus` |
| ≥2 个 | `options_2_plus__attrs_1_2` | `options_2_plus__attrs_3_plus` |

购买选项指 goal 指定的颜色、尺码等要求，计数来自 `goal_options`，不是网页提供的所有候选按钮数。属性指 `attributes` 中需要核对的商品特征，原生 human goals 已过滤零属性任务；字段缺失或格式错误会报错。具有相同计数档位的任务跨商品大类共享 rho。价格和商品大类不再细分这六组，goal ID、任务内容与 train/dev/test 划分不受影响。

这六组表达“核对与选项操作结构相似”的假设，尚未验证组内难度或对错误前缀的敏感度一致，也不预设约束越多一定越难。仅用这些标注确定训练分组，不将标注字段额外加入 solver 观测。生成清单时 `task_type_counts` 会列出各 split 的全部六组及数量，包括空组；curriculum 仅为训练池中实际出现的组创建 rho 状态。

`--webshop-rho-grouping category` 保留原先商品大类分组：使用 goal 的 `category`，缺失时归入 `shopping`。分组选择保存在 `tasks.json` 的 `backend_options.rho_grouping`，具体映射保存在各任务的 `task_type`；训练直接读取清单。旧清单继续使用原有分类，需重新生成清单才能启用六组。分组变化会改变环境/任务池指纹，不能直接恢复旧分组的 curriculum 检查点。

每批抽取 16 个具体任务，不要求是 16 种任务类型，也不保证各类型均衡。本轮出现的类型根据这些任务的平均成功率更新 rho；未出现的类型保持原值。分组规则保存在 manifest 中，修改分组后应重新生成任务清单。

## 推荐参数

环境参数来源是 [launch.py](launch.py)，DenoiseRL 的 rho 控制参数直接继承与 ALFWorld 共用的 [denoise_v2_base.yaml](../config/denoise_v2_base.yaml)，不再单独覆盖；所有额外 `key=value` 参数都原样传递给 Hydra，覆盖默认值。

| 参数 | WebShop baseline / denoise | ScienceWorld baseline / denoise |
|---|---:|---:|
| task batch size | 16 / 16 | 16 / 16 |
| 每任务 solver rollout | 16 / 16 | 16 / 16 |
| solver 轨迹 / step | 256 / 256 | 256 / 256 |
| 最大环境动作数（含前缀） | 15 | 100 |
| history length | 2 | 4 |
| prompt / response tokens | 4096 / 256 | 8192 / 256 |
| learning rate | 1e-6 | 1e-6 |
| KL loss coefficient | 0.01 | 0.01 |
| PPO mini / micro per GPU | 128 / 4 | 128 / 4 |
| rollout TP | 2 | 2 |
| 训练 temperature / top-p | 1.0 / 1.0 | 1.0 / 1.0 |
| 评估 | greedy，n=1 | greedy，n=1 |
| 训练 steps / dev 间隔 | 500 / 25 | 500 / 25 |
| 初始 rho | 0 / 0 | 0 / 0 |
| rho 范围 | 固定 0 / [0, 0.5] | 固定 0 / [0, 0.5] |
| rho alpha | 0 / 0.2 | 0 / 0.2 |
| target success | — / 0.75 | — / 0.75 |
| solver / weak vLLM memory fraction | 0.5 / 0.2（仅 denoise） | 0.5 / 0.2（仅 denoise） |
| 并行评估任务 | 16 | 8 |

这里 PPO mini batch 的单位是 collector 展开的动作行，不是完整轨迹。500 steps 是初次实验预算，不保证收敛；全量 dev 评估在 ScienceWorld 上可能较慢，可以增大 `trainer.test_freq`，不要通过减少 test 覆盖率来加速最终评估。

三个环境统一使用每批 16 个具体任务、每任务 16 条 rollout。DenoiseRL 的 initial_rho=0、min_rho=0、max_rho=0.5、target_accuracy=0.75、alpha=0.2；baseline 固定 rho=0。ScienceWorld 仅保留适合长任务的步数预算和历史长度。若 rho 长期为 0，应先检查对应任务类型的完全成功率。

WebShop 每 16 个逻辑环境共用一个 Ray actor 内的商品库/搜索索引，浏览器 session 分离；16 个训练任务对应 16 份商品库，而非 256 份。ScienceWorld 每条轨迹仍有独立 JVM（默认 256 个训练 JVM＋8 个验证 JVM）。CPU/RAM 受限时，把两个方法的 `TRAIN_BATCH_SIZE` 一起调小，并重新生成对应大小的 train.parquet。

## 安装与任务清单

WebShop 与 ScienceWorld 分别使用独立的 Python 3.10 Conda 环境；每个环境同时支持 baseline 和 DenoiseRL。安装器支持从零创建和从 `molu` 克隆，所有安装显式指向新环境，不修改源环境。克隆前和安装后的 `pip check` 均只记录警告，不阻止流程；实际安装、核心包导入和模拟器检查失败仍会停止。详见 [环境创建、数据准备与检查](ENVIRONMENTS.md)。

Conda 与 pip 默认使用内网 `nexus.sii.shaipower.online` 源，无需额外配置；pip 默认端点为 `pypi_proxy/simple/`，自动添加 trusted-host 和 120 秒超时。`--pip-index-url` 可改用内网的 `pypi/simple/`。`--mirror tuna` / `--mirror official` 可显式选择清华 / 官方源，也可在 `--resume` 续装时切换，详见 [下载源设置](ENVIRONMENTS.md#默认下载源)。

```bash
# 在 Linux GPU 服务器的仓库根目录创建。
python -m recipe.denoise_v2.task_suite.setup_environment --benchmark webshop --mode fresh
python -m recipe.denoise_v2.task_suite.setup_environment --benchmark scienceworld --mode fresh
```

WebShop 需要完整商品、属性、human instructions、spaCy 小模型和搜索索引。旧 `setup.sh` 默认禁用，不再用于这套安装流程。下面的数据脚本会复用已下载文件，在临时目录建完整索引，并保留旧索引备份；不会安装 Python/Conda 包。1000 商品集不用于标准划分。

数据下载默认使用固定版本的 Hugging Face 完整副本，并校验大小与 SHA256。Google Drive 原链接获取失败时无需重装环境；可用 `--hf-endpoint https://hf-mirror.com` 选择 HF 镜像，具体来源、续传与离线复制步骤见 [数据准备文档](ENVIRONMENTS.md#2-准备数据与任务清单)。

ScienceWorld 固定 [官方仓库](https://github.com/allenai/ScienceWorld) 包含确定性 reset 改进的源码提交，版本见 [依赖文件](requirements-scienceworld.txt)。安装器同时安装 Java 11。`prepare_tasks` 保存实际 Python 包版本和 JAR SHA256，运行时不匹配会报错；baseline/denoise 保持同一版本。

ScienceWorld 安装包包含模拟器 JAR 和任务定义，variation 在本地生成，无需另下商品式数据集。`prepare_tasks` 只枚举已安装的本地环境、写出清单与 parquet，不负责下载模型或原始数据。

```bash
conda activate denoise-webshop
python -m recipe.denoise_v2.task_suite.prepare_webshop_assets --download --source huggingface --build-index

# 从仓库根目录运行；仅发现任务及生成本地 parquet，不运行训练。
python -m recipe.denoise_v2.task_suite.prepare_tasks \
  --benchmark webshop --train-batch-size 16 \
  --webshop-rho-grouping structure \
  --output recipe/denoise_v2/local_data/webshop

conda activate denoise-scienceworld
python -m recipe.denoise_v2.task_suite.prepare_tasks \
  --benchmark scienceworld --train-batch-size 16 \
  --output recipe/denoise_v2/local_data/scienceworld
```

每个输出目录包含 `tasks.json`、`train.parquet`、`dev.parquet`、`test.parquet`。train.parquet 只有一个占位 batch，实际任务顺序由 curriculum 控制；配置中的 trainer epoch 与遍历完整任务池的 pool epoch 不同。

若比较原先的商品大类分组，单独生成清单，并为 baseline/denoise 选用相同数据目录；使用独立实验名保存结果：

```bash
python -m recipe.denoise_v2.task_suite.prepare_tasks \
  --benchmark webshop --webshop-rho-grouping category \
  --output recipe/denoise_v2/local_data/webshop_category

TASK_DATA_DIR=recipe/denoise_v2/local_data/webshop_category \
EXPERIMENT_NAME=webshop_category_denoise_7b_seed0 \
  bash recipe/denoise_v2/run_webshop_denoise_train.sh
```

新生成清单默认推荐 `structure`。baseline 读取相同清单但 rho 仍固定为 0；分组改动不改变每批任务数、rollout 数或 rho 更新公式。

ScienceWorld 默认 **不使用 simplifications**。若实验需要可显式传入 `--simplifications teleportAction,openDoors`，并为该协议使用独立的数据目录和实验名。可传 `--jar-path /absolute/scienceworld.jar` 指定 JAR。始终 `generateGoldPath=False`，不读取专家轨迹。

WebShop 可用 `--webshop-data-dir` 指定 JSON 文件目录，但搜索索引仍由 bundled WebShop 加载，须与这批商品匹配。manifest 记录绝对路径，迁移到集群后应在集群重新生成 manifest。

先检查原生环境的确定性重放：

```bash
conda activate denoise-webshop
python -m recipe.denoise_v2.task_suite.smoke_env \
  --benchmark webshop --manifest recipe/denoise_v2/local_data/webshop/tasks.json
conda activate denoise-scienceworld
python -m recipe.denoise_v2.task_suite.smoke_env \
  --benchmark scienceworld --manifest recipe/denoise_v2/local_data/scienceworld/tasks.json
```

## 训练

```bash
# WebShop
conda activate denoise-webshop
bash recipe/denoise_v2/run_webshop_grpo_train.sh
bash recipe/denoise_v2/run_webshop_denoise_train.sh

# ScienceWorld
conda activate denoise-scienceworld
bash recipe/denoise_v2/run_scienceworld_grpo_train.sh
bash recipe/denoise_v2/run_scienceworld_denoise_train.sh
```

默认模型名按 Hugging Face ID 解释。离线集群可设置 `MODEL_ROOT`，或直接设置绝对路径：

```bash
MODEL_PATH=/models/Qwen2.5-7B-Instruct \
DENOISE_MODEL_PATH=/models/Qwen2.5-1.5B-Instruct \
bash recipe/denoise_v2/run_scienceworld_denoise_train.sh \
  --seed 1 env.denoise.v2.target_accuracy=0.75
```

首次使用需准备两个 Hugging Face 模型的完整权重、tokenizer 与配置；已用于 ALFWorld 的本地模型可直接复用。baseline 仅加载 7B solver，DenoiseRL 还加载 1.5B 弱模型。可在联网机器上用训练环境已有的 `huggingface_hub` 下载，替换下面的存储目录为实际路径：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
for name in ("Qwen2.5-7B-Instruct", "Qwen2.5-1.5B-Instruct"):
    snapshot_download(repo_id=f"Qwen/{name}", local_dir=f"/data/models/{name}")
PY
```

原始数据、搜索索引、依赖和模型准备齐后可以离线训练。复制到集群后重新生成任务清单以写入目标机器上的绝对路径；指定本地 `MODEL_PATH`/`DENOISE_MODEL_PATH`，并设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`WANDB_MODE=offline`。两个模拟环境均在本地运行，WebShop 训练不访问真实电商网站。

WebShop/ScienceWorld 启动器的 rho 参数通过 Hydra 的 `env.denoise.v2.*=...` 覆盖；ALFWorld shell 脚本的 `V2_ALPHA`、`V2_MAX_RHO` 等环境变量不适用于这两个入口。例如：

```bash
TRAIN_BATCH_SIZE=16 N_GPUS_PER_NODE=8 \
bash recipe/denoise_v2/run_webshop_denoise_train.sh --seed 0 \
  env.denoise.v2.initial_rho=0.0 env.denoise.v2.min_rho=0.0 \
  env.denoise.v2.max_rho=0.5 env.denoise.v2.alpha=0.2 \
  env.denoise.v2.target_accuracy=0.75 \
  trainer.total_training_steps=500 trainer.total_epochs=500
```

支持 `TRAIN_BATCH_SIZE`、`VAL_BATCH_SIZE`、`N_GPUS_PER_NODE`、`EXPERIMENT_NAME`、`TASK_DATA_DIR`、`CKPT_DIR` 环境变量。其他参数使用 Hydra override。示例：

```bash
bash recipe/denoise_v2/run_webshop_denoise_train.sh --dry-run

# 查看将传给 Hydra 的完整参数列表，不启动模型/环境：
python -m recipe.denoise_v2.task_suite.launch --benchmark webshop --dry-run

# 短训练验证（仍需完整环境及 GPU）
EXPERIMENT_NAME=webshop_denoise_smoke \
bash recipe/denoise_v2/run_webshop_denoise_train.sh \
  trainer.total_training_steps=2 trainer.total_epochs=2 \
  trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=1
```

`--dry-run` 输出最终参数列表，不启动 Ray 或加载数据，不执行 Hydra 解析。`WANDB_MODE` 默认 offline。checkpoint 自动恢复默认开启；重做实验应使用新的 `EXPERIMENT_NAME` 或传 `trainer.resume_mode=disable`。恢复训练需要 `denoise_v2_curriculum.json`，其中记录任务顺序、rho、随机状态和环境指纹。

## 评估

```bash
# 基础模型（不允许缺少 checkpoint 时默默回退）
bash recipe/denoise_v2/run_webshop_eval.sh --base-model --eval-split test

# 两种训练方法共用同一个干净评估入口
bash recipe/denoise_v2/run_webshop_eval.sh --method baseline \
  --checkpoint checkpoints/denoise_v2_webshop/webshop_baseline_7b_seed0 \
  --eval-split test
bash recipe/denoise_v2/run_webshop_eval.sh \
  --checkpoint checkpoints/denoise_v2_webshop/webshop_denoise_7b_seed0 \
  --eval-split test
bash recipe/denoise_v2/run_scienceworld_eval.sh \
  --checkpoint checkpoints/denoise_v2_scienceworld/scienceworld_denoise_7b_seed0 \
  --eval-split test
```

`--checkpoint` 接受实验根目录、`global_step_N` 或其 `actor` 子目录。评估指标包括 `val/test/success_rate`（完全成功）、`val/test/test_score`（归一化最终分数）、各任务类型成功率及 `task_coverage=1`。ScienceWorld score 乘 100 才是官方 0–100 量纲。

rollout 和 validation JSONL 位于 `recipe/denoise_v2/dumps/<experiment>/`；validation 每行是一条折叠后的 solver 轨迹，包含实际 `task_id`。最终比较建议使用同一 manifest、相同种子集合，报告 SR、score、solver tokens、弱模型 tokens、环境交互数与墙钟时间；相同训练 steps 并不代表 DenoiseRL 的总计算量与 baseline 相同。

## 验证范围

CPU 契约测试覆盖固定任务、跨 split 去重、共享前缀、零 rho 快速路径、终止保护、奖励只支付一次、重放步数预算、baseline 任务遍历、checkpoint 指纹，以及评估任务覆盖率：

```bash
python -m unittest tests.recipe.test_task_suite tests.recipe.test_denoise_v2 \
  tests.recipe.test_denoise_v2_efficiency tests.recipe.test_alfworld_exhaustive_validation \
  tests.recipe.test_task_suite_environment
```

这些测试使用模拟 backend，不能替代上面的真实环境 smoke test 或 GPU 训练。推荐参数不是已经复现出的 benchmark 最优参数。
