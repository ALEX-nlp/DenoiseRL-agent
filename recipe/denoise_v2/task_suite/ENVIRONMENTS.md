# WebShop 与 ScienceWorld 的独立环境

两个 benchmark 各用一个 Conda 环境；同一 benchmark 的 GRPO baseline 与 DenoiseRL v2 共用环境，保证比较时依赖一致。两个环境均使用 Python 3.10，**独立环境不意味着必须使用不同 Python 版本**。

| 环境名 | 用途 | 模拟器依赖 |
|---|---|---|
| `denoise-webshop` | WebShop baseline / DenoiseRL | Java 11、Pyserini 0.17、spaCy 3.7.5、文本模拟器 |
| `denoise-scienceworld` | ScienceWorld baseline / DenoiseRL | Java 11、固定源码版本的 ScienceWorld 及其 JAR |

安装器适用于 Linux x86_64 的 NVIDIA GPU 训练机器。以下命令均在服务器的仓库根目录执行。原来的 `molu` 不作为安装目标。

## 1. 创建环境：选择一种方式

### 推荐：从零创建

如果 `molu` 仍有此前 `pip check` 中的冲突，采用此方式。Conda / pip 会复用已有下载缓存，但新环境不会继承 `molu` 的 Python 包。

```bash
python -m recipe.denoise_v2.task_suite.setup_environment --benchmark webshop --mode fresh
python -m recipe.denoise_v2.task_suite.setup_environment --benchmark scienceworld --mode fresh
```

两者使用相同的训练核心版本：

| 组件 | fresh 版本 |
|---|---|
| Python / Java | 3.10 / 11 |
| PyTorch / vLLM | 2.6.0 / 0.8.5.post1 |
| FlashAttention | 2.7.4.post1 |
| Transformers / Tokenizers | 4.51.3 / 0.21.1 |
| PEFT / Ray | 0.15.2 / 2.43.0 |
| TensorDict / TorchData | 0.8.3 / 0.11.0 |
| NumPy | 1.26.4 |
| OpenTelemetry API / SDK | 1.26.0 / 1.26.0 |

版本来源是 [训练约束](envs/train-constraints.txt) 和 [训练依赖](envs/train-requirements.txt)。这是针对 Qwen2.5 文本训练的固定核心版本组合，不是 `molu` 的完整复刻，也不是全部传递依赖的锁文件。安装完成会保存实际的完整版本快照。

默认 PyTorch 2.6 Linux wheel 使用 CUDA 12.4 运行库；服务器需要兼容的 NVIDIA 驱动。FlashAttention 安装可能需要下载匹配的预编译 wheel；若回退到源码构建，则需要匹配的 CUDA Toolkit / `nvcc` 和 C++ 编译器。可设置 `MAX_JOBS=4` 限制编译并行数。安装器不会安装系统驱动或 CUDA Toolkit。

### 快速方式：从健康的 molu 克隆

如果 `molu` 已恢复且训练正常，可以复用训练栈，省去重新安装 Torch / vLLM / FlashAttention：

```bash
python -m recipe.denoise_v2.task_suite.setup_environment \
  --benchmark webshop --mode clone --source-env molu
python -m recipe.denoise_v2.task_suite.setup_environment \
  --benchmark scienceworld --mode clone --source-env molu
```

克隆前会只读运行源环境的 `pip check`；有冲突就停止，建议改用 fresh。源环境必须是 Python 3.10，且已安装脚本要求的训练核心包。克隆使用 `conda create --clone ... --copy`，随后所有安装都显式指定新环境的 prefix。Torch、vLLM、FlashAttention、Transformers、PEFT、Ray 等以及已安装的 NVIDIA 包会记录为精确约束，补装模拟器时不能改动这些版本。其他模拟器依赖可在新环境中调整；冲突不会通过自动重装源环境来解决。

`--name my-webshop` 可自定义新环境名；不允许 `base`、`molu` 或源环境名。已有环境默认拒绝覆盖。

### 预览、失败续装与检查记录

```bash
# 只打印命令，不调用 Conda、不安装包；也可在非 Linux 机器预览。
python -m recipe.denoise_v2.task_suite.setup_environment --benchmark webshop --mode fresh --dry-run

# 仅续装由此脚本创建、且参数一致的目标环境。
python -m recipe.denoise_v2.task_suite.setup_environment --benchmark webshop --mode fresh --resume
```

安装完成前会执行 `pip check`、训练入口和核心包导入、Java 检查；WebShop 还检查文本模拟器及 spaCy 模型，ScienceWorld 会启动 JVM 并读取任务类型。记录位于新环境的 `$CONDA_PREFIX/.denoise-task-suite/`：

- `setup.json`：创建方式、源环境和安装状态。
- `pip-freeze.txt`、`check.json`：实际版本与依赖/导入检查结果。
- `core-constraints.txt`：clone 模式的原始训练核心约束，失败重试时不会重新生成以掩盖版本变化。

安装检查不等于 GPU 训练验证；继续执行下面的原生模拟器 smoke test 和短训练。

## 2. 准备数据与任务清单

### WebShop

```bash
conda activate denoise-webshop

# 已下载的完整 JSON 会复用；在临时目录重建完整搜索索引，成功后保留旧索引备份。
python -m recipe.denoise_v2.task_suite.prepare_webshop_assets --download --build-index

python -m recipe.denoise_v2.task_suite.prepare_tasks \
  --benchmark webshop --train-batch-size 16 --webshop-rho-grouping structure \
  --output recipe/denoise_v2/local_data/webshop

python -m recipe.denoise_v2.task_suite.smoke_env \
  --benchmark webshop --manifest recipe/denoise_v2/local_data/webshop/tasks.json
```

数据脚本仅下载和建索引，不执行任何 pip / conda 安装。文件保存在 bundled WebShop 的 `data/`，索引在 `search_engine/indexes/`；它们与模型缓存均可在不同 Conda 环境间复用。此前完整下载成功的 `items_shuffle.json`、`items_ins_v2.json`、`items_human_ins.json` 会直接使用，已有文件须完整有效。第一次迁移仍建议重建一次索引，避免沿用旧脚本生成的小商品集索引。失败的下载保留 `.partial`，不会替代目标文件。

仅需要 `en_core_web_sm`，已随依赖安装；不下载 `en_core_web_lg`。不需要启动 Flask 服务或真实浏览器。旧 `webshop/setup.sh` 默认禁用，防止再次修改当前环境；仅显式设置 `WEBSHOP_ALLOW_LEGACY_INSTALL=1` 才能运行旧流程，新训练流程无需该开关。

### ScienceWorld

```bash
conda activate denoise-scienceworld

python -m recipe.denoise_v2.task_suite.prepare_tasks \
  --benchmark scienceworld --train-batch-size 16 \
  --output recipe/denoise_v2/local_data/scienceworld

python -m recipe.denoise_v2.task_suite.smoke_env \
  --benchmark scienceworld --manifest recipe/denoise_v2/local_data/scienceworld/tasks.json
```

[ScienceWorld 依赖文件](requirements-scienceworld.txt) 固定官方源码提交 `e8216d6044e8e39be9fcb185e3b2dfb602584b52`，包含官方针对确定性 reset 的修改与模拟器 JAR；不是未固定的 `main` 或单纯安装 PyPI 1.2.3。任务 variation 在本地生成，无需额外下载商品式数据。清单记录实际包版本与 JAR SHA256；更换环境/JAR 后重新生成清单，不直接续接旧 curriculum checkpoint。

## 3. 启动训练

```bash
conda activate denoise-webshop
bash recipe/denoise_v2/run_webshop_grpo_train.sh
bash recipe/denoise_v2/run_webshop_denoise_train.sh

conda activate denoise-scienceworld
bash recipe/denoise_v2/run_scienceworld_grpo_train.sh
bash recipe/denoise_v2/run_scienceworld_denoise_train.sh
```

分别选择要运行的命令；每个训练进程默认占用 8 张 GPU，不应在同一组 GPU 上同时启动两个实验。baseline 与 DenoiseRL 的任务清单、solver 和 GPU 栈在各自 benchmark 内保持一致。

默认配置仍是 7B solver + 1.5B 固定弱模型、每批 16 个任务 × 每任务 16 条 solver 轨迹；baseline 不加载弱模型。DenoiseRL 的 rho 初始为 0，范围 `[0, 0.5]`，alpha=0.2，目标成功率 0.75。WebShop 默认六个结构组，ScienceWorld 按 task name 共享 rho。完整参数与评估命令见 [任务文档](README.md)。

先用一个独立实验名跑两步，例如：

```bash
EXPERIMENT_NAME=webshop_env_smoke \
bash recipe/denoise_v2/run_webshop_denoise_train.sh \
  trainer.total_training_steps=2 trainer.total_epochs=2 \
  trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=1
```

ScienceWorld 同样使用对应脚本和独立实验名。两步 smoke 主要检查训练链路；初始 rho=0，通常还不能验证非零前缀下的弱模型生成。ScienceWorld 默认会创建 256 个训练 JVM，需要相应 CPU / RAM；如果降低 `TRAIN_BATCH_SIZE`，两种方法一起调整并重建对应 batch size 的任务清单。

## 联网与验证范围

首次安装需访问 Conda channels、PyPI、GitHub（ScienceWorld、spaCy 模型，以及可能的 FlashAttention wheel）；WebShop 数据来自 Google Drive。模型权重来自 Hugging Face，已有 ALFWorld 的本地 Qwen2.5 模型可通过 `MODEL_PATH` / `DENOISE_MODEL_PATH` 复用，不必重复下载。依赖、JAR、数据、索引和模型齐备后可以离线运行，相关变量见 [任务文档](README.md)。

开发时已针对 Linux x86_64 / Python 3.10 解析两个 fresh profile 的依赖；FlashAttention 单独安装，尚未验证其编译与 CUDA ABI。CPU 测试覆盖安装目标隔离、失败恢复和数据索引替换。未在本地 macOS 上创建这些 CUDA 环境，服务器上的完整安装、原生环境 smoke test 和 GPU 短训练仍需实际执行。
