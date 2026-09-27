**ScienceWorld 参考实现与本项目对照（2026-09-27）**

分析对象是用户提供的 `/Users/molu/Downloads/sciworld_design.zip`，重点为 `qwen3_5_4b_sciworld_grpo_160step_four_gpu_baseline_batch_24_task_balance`，并单独说明同包中的 conflict 实验。压缩包已解到 `/tmp/sciworld_reference_audit_20260927`。本次仅静态检查源码、配置、资产元数据及清单；没有运行上传的脚本、启动训练、重放 JVM 或修改本项目训练配置。

“我们当前”指本地 launcher 的默认配置，包含此前刚改好的 ScienceWorld 成功奖励及采样评测。它不代表已经启动的远端 W&B 作业自动采用了这些更改。命令行和环境变量仍可覆盖默认参数。

最重要的结论：参考实验是 **L2 自定义划分、15 次交互、固定 64 题的短预算 GRPO 实验**。它不是本项目所采用的全 30 类、270 个 test 实例、300 原生 moves / 600 次动作的正式评测协议。不能根据它使用 15 步，就认定我们 50 步足够；也不能直接比较两边同名 score。

**实际启用的执行链**

1. `qz_launch.py` 读取 `source_config.json`，重定位输出与资产路径，将 batch 改为 24、总训练更新数改为 160，禁用 resume，从初始 Qwen3.5-4B 开始。名称中的 160 是训练更新次数，不是单条轨迹长度。
2. `prepare --mode text` 生成训练器使用的占位数据；实际任务由环境从 `L2_idx.json` 的 train 池采样。`train_batch_size=24` 不表示只有 24 个训练任务。
3. `run_resume_config.py` 将完整配置送入正常 PPO driver。算法实际为 GRPO，不是 GiGPO；配置保留的 GiGPO 参数没有因此启用。
4. `SciWorldEnvironmentManager` 使用原生 ScienceWorld worker。每批选 24 个任务实例，同一个实例复制 8 份，形成 192 条轨迹。`rollout.n=1` 并不意味着每题只采一条，重复发生在 `env.rollout.n=8`。
5. 自定义 `multi_turn_rollout/rollout_loop.py` 最多运行 15 次模型—环境交互。因此配置中的 `rollout.multi_turn.enable=false` 不代表它只执行单轮。
6. 每 5 次训练更新，评估相同的 64 个 L2/test 实例；greedy、每题 1 条、同样最多 15 次交互。没有训练前第 0 步评估。每 10 次更新保存 checkpoint。

证据：[launcher](/tmp/sciworld_reference_audit_20260927/experiments/sciworld/qwen3_5_4b_sciworld_grpo_160step_four_gpu_baseline_batch_24_task_balance/qz_launch.py:41)、[实际有效配置](/tmp/sciworld_reference_audit_20260927/experiments/sciworld/analysis_0924/raw_data/baseline/effective_config.json)、[环境构建](/tmp/sciworld_reference_audit_20260927/agent_system/environments/env_manager.py:5463)、[轨迹循环](/tmp/sciworld_reference_audit_20260927/agent_system/multi_turn_rollout/rollout_loop.py:340)。

**逐项对照**

| 项目 | 参考 baseline160 / batch24 | 我们本地 ScienceWorld 默认配置 |
| --- | --- | --- |
| Solver | Qwen3.5-4B | Qwen2.5-7B-Instruct |
| 方法 | clean GRPO | clean GRPO / DenoiseRL v2 |
| Denoise | baseline 没有弱模型前缀 | denoise 使用 Qwen2.5-1.5B-Instruct 生成前缀，同组 solver 共享前缀；baseline 的 rho 固定为 0 |
| 训练任务来源 | 自定义 `L2_idx.json['train']` | 原生官方 train split，覆盖 30 类 |
| 训练采样 | task-balanced：类别轮转配额，类内随机选 variation | 对完整实例池按 epoch 打乱遍历，每次更新换新实例；不是类别等权 |
| 每批轨迹 | 24 × 8 = 192 | 16 × 8 = 128 |
| 训练动作预算 | 外层 15；原生 `envStepLimit=15` | 外层 50；原生 50；denoise 重放前缀也计入预算 |
| 成功奖励 | 原生 score ≥ 100 给 10，否则 0 | 终止时成功给 1，否则 0 |
| 非法惩罚 | −0.1，按输出格式判断 | −0.1，检查动作格式及原生可执行动作集合 |
| 简化参数 | 默认传入 `easy` | 显式传入 `easy` |
| 训练采样解码 | temperature=0.4，top_p=1，top_k=−1 | temperature=1，top_p=1，top_k=−1 |
| 评测解码 | greedy，temperature=0，top_p=1，n=1 | temperature=0.6，top_p=0.95，top_k=−1，n=1 |
| 单次生成上限 | 1024 tokens | 256 tokens |
| Prompt 上限 | 6000 tokens | 8192 tokens |
| 历史 | 最近 2 条详细观察及动作，较早动作名也保留 | 最近 4 条观察及动作；当前状态包含 observation、look、inventory |
| 在线评测 | 每 5 更新，同一份 64 个 L2/test 实例，15 步 | 每 25 更新，官方 dev 每类最多 3 个实例，50 步 |
| 最终 test | 本次入口继续使用同一 eval64；未见自动切到 270 题正式协议 | 默认最终 checkpoint 评估 270 个 test 实例；600 动作 / 300 moves，停滞终止 |
| 优化 | GRPO，lr=1e−6，actor KL=0.01，k2 | GRPO，lr=1e−6，actor KL=0.01，low_var_kl |
| 熵 / PPO epoch | 0.001 / 1 | 0.001 / 1（继承基础配置） |
| PPO minibatch / microbatch | 64 / 每 GPU 1 | 128 / 每 GPU 4 |
| GPU / vLLM TP | 4 / 1 | 默认 8 / 2 |
| 训练总更新数 | 160 | 默认 500 |

动作次数与原生 moves 不是同一个量；原生 wrapper 使用 `moves > envStepLimit` 的结束条件。某些查询/无效动作不推进原生 moves，`wait` 可推进多个时间步；外层循环仍单独限制模型调用次数。

双方的 `use_kl_in_reward=false`，不能把配置里的 `algorithm.kl_ctrl.kl_coef=0.001` 当成本次实际 actor KL 系数。双方都使用 episode outcome，展开成各动作记录，再做 GRPO；本次检查的核心优势估计代码都默认跨组内动作记录计算均值/标准差，并不是此参考实验特有的算法差异。

我们的来源：[launcher](/Users/molu/Documents/DenoiseRL-agent/recipe/denoise_v2/task_suite/launch.py:73)、[任务池遍历](/Users/molu/Documents/DenoiseRL-agent/recipe/denoise_v2/gamefile_curriculum.py:29)、[配置继承](/Users/molu/Documents/DenoiseRL-agent/recipe/denoise_v2/config/denoise_v2_base.yaml:1)、[评测说明](/Users/molu/Documents/DenoiseRL-agent/recipe/denoise_v2/task_suite/SCIENCEWORLD_EVALUATION.md)。

**奖励目标一致，但尺度及非法动作定义不同**

参考代码直接用 `10 * (score >= 100)` 替换原生增量奖励。超时、部分完成、后端负分失败都给 0；不是看到 `done` 就算成功。保存到每个活跃动作记录的 episode 回报用于 GRPO，再对格式非法的该动作减 0.1。`reward.py` 虽然有 `native_final` 部分分奖励，但实际配置选的是 `strict_success`，不能用未启用分支解释本次实验。

我们已采用同样的成功/失败目标，但成功尺度为 1。单纯将所有奖励统一乘 10，在组内标准化 GRPO 下通常大致抵消；这里惩罚固定为 0.1，因此并非完全等价：它相当于参考成功回报的 1%，而是我们成功回报的 10%。不能直接说策略梯度因此放大或缩小十倍。

参考投影器只检查完整动作标签、存在 `</think>`、没有中文等格式条件。动作是否位于环境合法动作集合另外记录为 `action_available`，没有进入该惩罚判断。我们要求恰好一个非空单行动作标签，再结合原生合法动作集合判断。两边的 `valid_action_ratio` 因此也不能直接横向比较。

证据：[参考奖励](/tmp/sciworld_reference_audit_20260927/agent_system/environments/env_package/sciworld/envs.py:16)、[参考投影](/tmp/sciworld_reference_audit_20260927/agent_system/environments/env_package/sciworld/projection.py:4)、[参考惩罚](/tmp/sciworld_reference_audit_20260927/verl/trainer/ppo/ray_trainer.py:200)、[我们的结算](/Users/molu/Documents/DenoiseRL-agent/agent_system/environments/env_package/task_suite/runtime.py:65)、[我们的有效性判断](/Users/molu/Documents/DenoiseRL-agent/agent_system/environments/env_package/task_suite/manager.py:80)。

**评测清单和指标是最需要区分的地方**

实际 `eval64.json` 的 SHA256 与 launcher 冻结值一致：`e2fc82497e9f6234fc754da52999f2d9c164fc674e73dda94d679ca918901887`。清单包含 64 个不重复的 `(task_num, variation)`；baseline 和 conflict 的清单逐字节相同。元数据记录：从 L2/test 的 549 个候选中用 Python `random.sample`、seed=42、不放回抽 64 个。

我直接统计到的类别分布如下，不是引用报告里的汇总值：

| task_num | 1 | 4 | 8 | 11 | 16 | 18 | 21 | 24 | 28 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 实例数 | 1 | 2 | 9 | 1 | 3 | 6 | 11 | 8 | 23 |

只覆盖 9 类，task28 和 task21 占 34/64 = 53.125%。训练 task-balanced，并不意味着评测也类别均衡。固定清单有利于跟踪同一批题，但其均值主要受高占比类别影响。64 题中每多成功一题，成功率改变 1.5625 个百分点。

`generalization_level=2` 在执行代码中表示选择 `L2_idx.json`，不是环境难度等级。附带的历史分析文字称该划分为原生 19 个训练类、10 个测试类，eval64 漏掉 task13；但 ZIP 没有 `L2_idx.json`，因此训练类集合、测试池的完整组成及 train/test 类别不相交性还不能独立复核。已能直接证实的是 eval64 的上述 9 类分布。

| 指标 | 参考实现含义 | 对照我们 |
| --- | --- | --- |
| `val/success_rate` | 每条 episode 最后一个活跃动作后的 score ≥ 100，64 题等权平均 | 对应完全成功率；我们使用 `val/dev/success_rate`、`val/test/success_rate` |
| `val/sciworld_native_score` | 最后一个活跃动作后的原生 score，episode 等权；失败 −100 也纳入 | 不等于我们的 `score_macro`，也不等于成功率 |
| `val/text/test_score` | episode 奖励复制到每个动作记录后再求平均 | 隐含按轨迹长度加权，不宜当成准确率 |
| 我们的 `val/<split>/score_macro` | 各任务类型先求末次非负进度均值，再对类型等权，范围 0–100 | 适合按本项目选定的 ScienceWorld 协议报告进度得分；名称应是分数，不是准确率 |

例如一条轨迹从 60 分走到原生失败 −100：参考 native 指标取 −100，我们末次非负分口径保留 60；两边成功率都为 0。即使换成完全相同的任务清单，score 仍可能相差很大。

汇报“成功率/准确率”时，应使用 `val/test/success_rate × 100%`；汇报 ScienceWorld 部分完成得分时，使用 `val/test/score_macro`，并写清任务清单、动作预算、解码参数、环境版本和分数规则。开发集与最终测试集结果需分开标记。

证据：[eval64](/tmp/sciworld_reference_audit_20260927/experiments/sciworld/analysis_0924/raw_data/baseline/eval64.json)、[episode 指标](/tmp/sciworld_reference_audit_20260927/agent_system/environments/env_manager.py:3222)、[参考验证聚合](/tmp/sciworld_reference_audit_20260927/verl/trainer/ppo/ray_trainer.py:834)、[动作加权 test_score](/tmp/sciworld_reference_audit_20260927/verl/trainer/ppo/ray_trainer.py:870)、[我们的宏平均](/Users/molu/Documents/DenoiseRL-agent/agent_system/scienceworld_protocol.py:32)。

**Prompt、思考预算和动作解析**

- 参考 prompt 要求先 `<think>…</think>` 再 `<action>…</action>`，提供任务描述、当前 observation、动作模板与可用物体列表。`meta_think=false`；普通 SciWorld manager 没有把 gold actions 或 goal progress 拼进 prompt。
- Worker 的确启用 `generateGoldPath=True` 并将 gold actions / goal progress 放进诊断 info；这不等于模型拿到了标准答案，也没有在当前 baseline 分支执行专家前缀。我们的 reset 不生成 gold path。
- 参考 `max_thinking_budget=800` 虽被读入，但 `ThinkLimitProcessor` 的构造和接入全部注释掉了。实际能确认的限制是单次回答总计 1024 tokens，不能声称为 action 预留了 224 tokens。
- 参考解析器取最后一对完整 action 标签；缺标签时把输出最后 20 个字符交给环境。长思考耗尽回答预算时，推理残片可能成为无效动作，并消耗外层交互预算。
- 我们允许简短思考但不强制 think 标签，拒绝多动作标签，解析失败转为显式无效动作。我们的回答预算只有 256；若未来换推理模型，需要单独验证是否足够。
- 双方都传 `easy`，未单独添加 `openContainers`。但 ZIP 没有实际 JAR 与简化规则的 Scala 实现，不能仅凭字符串就断言两个后端的每一项行为完全相同。

证据：[参考 prompt](/tmp/sciworld_reference_audit_20260927/agent_system/environments/prompts/sciworld.py:2)、[参考 manager](/tmp/sciworld_reference_audit_20260927/agent_system/environments/env_manager.py:3090)、[未接入的思考限制](/tmp/sciworld_reference_audit_20260927/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:327)、[我们的 prompt](/Users/molu/Documents/DenoiseRL-agent/agent_system/scienceworld_protocol.py:76)。

**同包 conflict 实验与 DenoiseRL 的区别**

conflict 配置额外指定 synthetic manifest 和 transfer JAR，原生任务仍切回 original JAR；评测始终使用 original JAR 和同一 eval64。生成脚本设计为增加 10 类合成任务、每类 10 个 variation，和原生任务一起按类别平衡采样；设计的合成占比为 10/29。这样在总 batch 不变时，会减少原生任务的训练暴露量。

这是改变训练任务分布/环境机制的实验，和我们“同一原生任务上重放弱模型前缀，再由 solver 恢复”的 denoise 并不相同。`construction.py` 自己标为归档 v1 原型；不能把其中的 Python 动作互换或专家前缀机制当成本次 backend-v2 conflict 的真实实现。

实际 `conflict_mixture.json`、`TaskTransfer.scala` 和 transfer JAR 不在 ZIP 中，所以各合成任务究竟改变了什么规则、是否可在 15 步内完成，本次无法验证。不能用未运行的原型或附带分析文字代替这些缺失实现。

证据：[conflict 有效配置](/tmp/sciworld_reference_audit_20260927/experiments/sciworld/analysis_0924/raw_data/conflict/effective_config.json)、[合成清单生成器](/tmp/sciworld_reference_audit_20260927/scripts/sciworld_env_construction/prepare_experiments.py:10)、[归档原型声明](/tmp/sciworld_reference_audit_20260927/agent_system/environments/env_package/sciworld/construction.py:1)、[JAR overlay 构建脚本](/tmp/sciworld_reference_audit_20260927/scripts/sciworld_env_construction/build_backend.py:38)。

**对我们当前配置的建议**

1. 保留成功奖励与进度评测分离，并同时报告完整成功率和 macro score。若要严格复现实验，应连同成功奖励尺度、非法惩罚定义一起对齐，不能只改 reward 名字。
2. 优先考虑 task-balanced 作为单独对照：我们的主要报告指标是类别宏平均，但当前训练按实例池遍历，variation 多的类别会得到更多训练机会。这是可借鉴的明确设计差异，不代表已经证明它会提高成绩。
3. 不直接把我们的训练预算改成 15，也不能凭这个 ZIP 判定应改为 100。先在相同 checkpoint、固定 dev 清单及相同解码下比较 50/100 步的成功率、macro score、截断率与有效进展，区分“少几步没做完”和“持续循环”。
4. 保留独立 dev 与最终 test。参考实现反复监控其名为 test 的 L2 子集；若据此调参或选模型，这个子集承担的是验证集角色，仍应另留未参与选择的最终测试数据。
5. 不将参考实验视为可靠性能上限：ZIP 没有原始训练/验证轨迹，`train_log_tails.txt` 为空。附带报告生成脚本包含历史成功率和预算耗尽统计，但本次没有独立复算这些数字，不能由它们解释我们当前远端 run 的异常。

**复核范围与缺失材料**

冻结启动记录中的 commit 是 `2732d7f295917dd71d0af830843aa76a3c5e0cd3`；ZIP 根目录记录的打包 checkout 为 `144fc65487fae215cb3db292c3082c3dada12500`。包内保留了 10 个关键训练/评测文件的启动 commit 副本。我逐一核对，它们与打包当前源码相同，且该次 baseline 启动 patch 未覆盖这些文件。因此上述核心行为不是把后续分支误当成启动时配置。

原版资产记录的 JAR SHA256 为 `acf78433198e5464659df7f6bdeb0503bc9b5802c10c27ebc5aa2eb9afe05e90`；Python 包 `version.py` 写 1.2.2，但不据此推定冻结 JAR 的构建版本。我们固定的是另一引擎提交并验证运行时签名，环境版本也需独立对齐。

若要进一步完整复现其设计，最优先补齐：`agent_system/environments/env_package/sciworld/variations_idx/L2_idx.json`；研究 conflict 时再补实际 `conflict_mixture.json` 及对应 `TaskTransfer.scala` / JAR 构建记录。要分析真实截断率和训练成效，还需要按 episode 保存的验证终态或原始轨迹及对应冻结 JAR。
