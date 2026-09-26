# WebShop 与 ScienceWorld 的独立环境

两个 benchmark 各用一个 Conda 环境；同一 benchmark 的 GRPO baseline 与 DenoiseRL v2 共用环境，保证比较时依赖一致。两个环境均使用 Python 3.10，**独立环境不意味着必须使用不同 Python 版本**。

| 环境名 | 用途 | 模拟器依赖 |
|---|---|---|
| `denoise-webshop` | WebShop baseline / DenoiseRL | Java 11、Pyserini 0.17、spaCy 3.7.5、文本模拟器 |
| `denoise-scienceworld` | ScienceWorld baseline / DenoiseRL | Java 11、固定源码版本的 ScienceWorld 及其 JAR |

安装器适用于 Linux x86_64 的 NVIDIA GPU 训练机器。以下命令均在服务器的仓库根目录执行。原来的 `molu` 不作为安装目标。

### 默认下载源

安装脚本默认使用内网 Nexus 源（`--mirror internal`），无需手动配置 `.condarc`、pip.conf 或执行 `export`：

- Conda 的 `pkgs/main`、`pkgs/r` 均使用 `http://nexus.sii.shaipower.online/repository/anaconda/pkgs/` 下对应地址；安装 Java 所需的 `conda-forge` 使用 `http://nexus.sii.shaipower.online/repository/anaconda/cloud/conda-forge`。通过 `--override-channels` 指定本次使用的频道。
- 所有 pip 安装命令默认使用 `http://nexus.sii.shaipower.online/repository/pypi_proxy/simple/`，并带上 `--trusted-host nexus.sii.shaipower.online --timeout 120`。
- 安装子进程忽略旧 pip 配置文件与环境变量中的索引、额外索引、find-links 和 trusted-host，避免混入公共备用源。网络代理环境变量仍可使用。
- 配置仅作用于脚本发起的安装命令，不修改用户全局 Conda / pip 配置；版本约束保持不变。Conda 使用用户提供的 HTTP 地址，不额外关闭全局 HTTPS 证书校验。

用户提供的另一个内网 pip 地址 `pypi/simple/` 可以显式选择；对应的 trusted-host 会自动添加：

```bash
python -m recipe.denoise_v2.task_suite.setup_environment \
  --benchmark scienceworld --mode fresh --resume \
  --pip-index-url http://nexus.sii.shaipower.online/repository/pypi/simple/
```

`--mirror tuna` 和 `--mirror official` 保留为显式选择，分别切换到清华和官方源。如果此前已进入依赖安装阶段，还可以同时加 `--resume`；切换 `--mirror` 或 `--pip-index-url` 不会触发环境身份不匹配：

```bash
python -m recipe.denoise_v2.task_suite.setup_environment \
  --benchmark webshop --mode fresh --resume --mirror official
```

`--dry-run` 会显示所用源的完整 URL。GitHub 上的 ScienceWorld、spaCy 模型及可能的 FlashAttention wheel，以及 Hugging Face 数据/模型不通过 Conda/pip 内网索引下载。WebShop 支持单独配置 HF 下载端点，见下文。克隆会复用本地缓存；Conda 若必须补下载源环境的精确包，可能仍访问该包记录中的原 URL。

## 1. 创建环境：选择一种方式

### 推荐：从零创建

如果希望从独立的依赖清单开始，可以采用此方式。Conda / pip 会复用已有下载缓存，但新环境不会继承 `molu` 的 Python 包。

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

### 快速方式：从 molu 克隆

克隆可以复用 `molu` 的训练栈，省去重新安装 Torch / vLLM / FlashAttention。`pip check` 的依赖冲突或平台标签提示不会阻止克隆：

```bash
python -m recipe.denoise_v2.task_suite.setup_environment \
  --benchmark webshop --mode clone --source-env molu
python -m recipe.denoise_v2.task_suite.setup_environment \
  --benchmark scienceworld --mode clone --source-env molu
```

克隆前会只读运行源环境的 `pip check`；非零退出码仅打印警告，随后继续克隆。源环境必须是 Python 3.10，且已安装脚本要求的训练核心包。克隆使用 `conda create --clone ... --copy`，随后所有安装都显式指定新环境的 prefix。Torch、vLLM、FlashAttention、Transformers、PEFT、Ray 等以及已安装的 NVIDIA 包会记录为精确约束，补装模拟器时不能改动这些版本。其他模拟器依赖可在新环境中调整；源环境不因这些检查而被修改。不要在外层 `set -e` 脚本中另加一条独立的 `python -m pip check`，否则仍会在进入安装器之前停止；若单独查看冲突，可用 `python -m pip check || true`。

`--name my-webshop` 可自定义新环境名；不允许 `base`、`molu` 或源环境名。已有环境默认拒绝覆盖。

### 预览、失败续装与检查记录

```bash
# 只打印命令，不调用 Conda、不安装包；也可在非 Linux 机器预览。
python -m recipe.denoise_v2.task_suite.setup_environment --benchmark webshop --mode fresh --dry-run

# 仅续装由此脚本创建、且参数一致的目标环境。
python -m recipe.denoise_v2.task_suite.setup_environment --benchmark webshop --mode fresh --resume
```

遇到 `Refusing to change existing environment` 或 `Environment ... already exists` 时，若此前运行的就是这个安装脚本，在原命令后加 `--resume`，保留原来的 `--benchmark`、`--mode` 和 `--source-env`。例如 ScienceWorld 的 fresh 安装续装命令为：

```bash
python -m recipe.denoise_v2.task_suite.setup_environment --benchmark scienceworld --mode fresh --resume
```

若提示 `installer record is missing`，说明目标环境缺少 `.denoise-task-suite/setup.json`，仅加 `--resume` 不能接管它。可能是在 Conda 创建结束、脚本写入记录之前中断，也可能是手动创建的环境。先检查目标环境与 Conda 安装历史，或通过 `--name` 使用另一个新环境；不要手工伪造安装记录。`--mode clone` 创建的环境应继续使用 clone 参数。

安装完成前会执行 `pip check`、训练入口和核心包导入、Java 检查；WebShop 还检查文本模拟器及 spaCy 模型，ScienceWorld 会启动 JVM 并读取任务类型。**安装后的 `pip check` 也只报告警告，不会阻止完成安装**；实际安装失败、核心包导入失败、Java/模拟器错误和核心版本约束变化仍会报错并停止。记录位于新环境的 `$CONDA_PREFIX/.denoise-task-suite/`：

- `setup.json`：创建方式、源环境、本次镜像选择、pip 索引和安装状态。
- `pip-freeze.txt`、`check.json`：实际版本与依赖/导入检查结果；`pip_check` 保存完整输出，`pip_check_returncode` 保存退出码，`warnings` 记录非阻断提示，`errors` 记录阻断错误。
- `core-constraints.txt`：clone 模式的原始训练核心约束，失败重试时不会重新生成以掩盖版本变化。

安装完成状态 `dependencies_checked` 不代表 `pip check` 零冲突，也不表示已自动修复全部冲突。安装检查不等于 GPU 训练验证；继续执行下面的原生模拟器 smoke test 和短训练。

## 2. 准备数据与任务清单

### WebShop

```bash
conda activate denoise-webshop

# 已下载的完整 JSON 会复用；在临时目录重建完整搜索索引，成功后保留旧索引备份。
python -m recipe.denoise_v2.task_suite.prepare_webshop_assets \
  --download --source huggingface --build-index

python -m recipe.denoise_v2.task_suite.prepare_tasks \
  --benchmark webshop --train-batch-size 16 --webshop-rho-grouping structure \
  --output recipe/denoise_v2/local_data/webshop

python -m recipe.denoise_v2.task_suite.smoke_env \
  --benchmark webshop --manifest recipe/denoise_v2/local_data/webshop/tasks.json
```

数据脚本仅下载和建索引，不执行任何 pip / conda 安装。默认来源已改为 [HongbangYuan/webshop 的固定版本](https://huggingface.co/datasets/HongbangYuan/webshop/tree/0129d4a81dbdb827e76afd20a1e2c38b61098613)，是社区托管的完整数据副本。三份文件的大小与校验值已和 [YWZBrandon/webshop-data](https://huggingface.co/datasets/YWZBrandon/webshop-data/tree/ce990fff5aee388db2706f07820c578ab68e0453) 交叉核对；大文件读取 HF LFS 元数据，小型人工指令文件另行下载计算 SHA256。未重新下载 Google Drive 原文件做逐字节比较。

`items_shuffle.json` 约 5.48 GB，`items_ins_v2.json` 约 186 MB，`items_human_ins.json` 约 5.14 MB。文件保存在 bundled WebShop 的 `data/`，索引在 `search_engine/indexes/`。下载先写入 `data/.hf-download/`，完成后流式校验大小与 SHA256，再移动到目标位置；HF 下载中断后可重跑相同命令继续。已有 JSON 校验通过后复用；校验不符会报出具体文件，不覆盖原文件。请给数据和索引的临时构建文件预留磁盘空间。

若服务器不能直连 Hugging Face，可使用 [HF-Mirror](https://hf-mirror.com/) 或其他可访问的兼容端点：

```bash
HF_HUB_DISABLE_XET=1 \
python -m recipe.denoise_v2.task_suite.prepare_webshop_assets \
  --download --source huggingface --hf-endpoint https://hf-mirror.com \
  --build-index --threads 4
```

也支持环境变量 `HF_ENDPOINT`。显式 `--hf-endpoint` 优先；下载公共数据时不发送已保存的 HF token。镜像连通性仍需在目标服务器验证，内网 pip 源不会代理 HF 或 Google Drive。Google Drive 保留为 `--source google-drive`，使用原始链接与 `.json.partial` 临时文件；原链接获取失败时建议改用 Hugging Face，无需重装环境。

如果只能在其他机器下载，可从上面的固定版本获取这三个 JSON 并复制到 bundled WebShop 的 `data/`，随后在训练服务器执行 `python -m recipe.denoise_v2.task_suite.prepare_webshop_assets --build-index`。第一次迁移建议重建完整索引，避免沿用旧脚本生成的小商品集索引。旧的 Google Drive `.partial` 不会被当成完整数据或拼接到 HF 下载。

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

首次安装默认通过内网 Nexus 源获取 Conda / PyPI 包，同时需要访问 GitHub（ScienceWorld、spaCy 模型，以及可能的 FlashAttention wheel）；这些固定直链没有对应的内网制品地址，不能仅靠包索引换源替代。WebShop 数据默认来自固定版本的 Hugging Face 副本，也支持原始 Google Drive 来源。训练启动器默认复用 `/inspire/hdd/global_user/xucaijun-253108120121/Model/Qwen/` 下的 `Qwen2.5-7B-Instruct` 和 `Qwen2.5-1.5B-Instruct`，不必重复下载；可通过 `MODEL_ROOT` 或 `MODEL_PATH` / `DENOISE_MODEL_PATH` 覆盖。依赖、JAR、数据、索引和模型齐备后可以离线运行，相关变量见 [任务文档](README.md)。

开发时已针对 Linux x86_64 / Python 3.10 解析两个 fresh profile 的依赖；FlashAttention 单独安装，尚未验证其编译与 CUDA ABI。CPU 测试覆盖安装目标隔离、失败恢复和数据索引替换。未在本地 macOS 上创建这些 CUDA 环境，服务器上的完整安装、原生环境 smoke test 和 GPU 短训练仍需实际执行。
