# ScienceWorld 训练监控与最终评测

本项目选择 **SwiftSage（NeurIPS 2023）的评测口径**作为参照。ScienceWorld 没有一个覆盖所有论文的统一运行脚本，不能只比较名字相同的 score。

## 可核对的参考实现

- [论文 Appendix A](https://arxiv.org/html/2305.17390v2#A1)：每类型保留前 10 个 test variation，少于 10 个的全部保留，共 270 个；不可恢复失败使用末次非负分数。
- [官方 eval_utils.py，science_world 分支](https://github.com/SwiftSage/SwiftSage/blob/science_world/eval_utils.py)：dev 取前 3 个；`test_mini` 取前三个，`test_mini_2` 取第 4–10 个，两者合起来对应论文 test。
- [官方 eval_agent_fast_slow.py](https://github.com/SwiftSage/SwiftSage/blob/science_world/eval_agent_fast_slow.py)：默认 `simplification_str=easy`、`env_step_limit=300`、外层动作循环上限为其两倍；累计 reward 历史长度达到 100 且最近 30 次 reward 之和为 0 时停止，并保留上一步分数。
- [本项目固定的 ScienceWorld 源码](https://github.com/allenai/ScienceWorld/tree/e8216d6044e8e39be9fcb185e3b2dfb602584b52)：使用官方 split，variation ID 按数值递增选取；环境 moves 计数不包含所有免费动作，所以另外限制总动作次数。原生 wrapper 在 moves **超过** 300 时结束，保留这个原生边界行为。

参考仓库的默认分支是另一版本，以上链接特意固定到 `science_world`。SwiftSage 原实验使用 scienceworld 1.1.3，本项目保留已固定的、修复确定性 reset 的引擎提交。因此这里是**对齐任务选择和评测规则**，不是声称逐位复现原论文；每次报告保存本地包版本、JAR SHA256、有效环境参数、任务 ID 清单和清单 SHA256。

## 默认流程

| 阶段 | 任务 | 每条轨迹预算 | 用途 |
|---|---|---|---|
| 训练 | 官方 train 池，每批 16 个任务、每题 8 条 solver 采样 | 50 次动作，含重放前缀 | GRPO / DenoiseRL 更新 |
| 每 25 个更新监控 | 每类型前 3 个 dev variation，约 90 个 | 50 次动作 / 50 原生 moves | 趋势监控，greedy |
| 训练后正式评测 | 每类型前 10 个 test variation，共 270 个 | 600 次动作 / 300 原生 moves，并使用停滞终止规则 | 报告最终结果，greedy |

默认跳过训练前评估；如需要初始基线，可单独运行基础模型评测。训练监控分数受较短预算影响，不应直接与正式 test 分数比较。baseline 与 DenoiseRL 使用相同环境设置、计分和动作格式；`easy` 与末次非负计分同时用于两个方法的训练与评估。原生 reward 在终止时只支付一次；完全成功仍按原生 score=100 统计。

训练预算 `16 × 8 × 50` 参考 [Paying Less Generalization Tax 附录 A.4](https://arxiv.org/html/2601.18217v1#A1.SS4)，用于降低在线训练成本；模型起点和奖励并不完全相同，不保证同样效果。每次更新的 solver 动作上限从原 `16 × 16 × 100 = 25,600` 降为 `6,400`，真实耗时仍受推理和并行效率影响，弱模型生成开销另计。

50 turn 是训练折中，并不足以保证覆盖所有长任务。[SwiftSage Appendix A](https://arxiv.org/html/2305.17390v2#A1) 列出的长任务专家轨迹平均长度可超过 100 步；具体版本、简化设置和策略都会改变动作数。先观察 `episode/truncation_rate` 与 `val/dev/truncation_rate`（0–1）、`truncated_score_mean`（仅存在截断时输出，0–100）及无效动作情况。若大量有效、有进展的轨迹被截断，可在同一固定 dev 集上对比 50/100 步，再决定是否延长训练预算。最终评测仍使用原有 300 moves / 600 动作，避免用短预算冒充正式结果。

截断指标区分预算耗尽与成功、不可恢复失败、停滞终止：只有动作上限或原生 moves 上限结束的未完成轨迹计入截断。重放前缀计入 50 步总预算。成功恰好发生在第 50 步时不计入截断，终止后重复调用不重复计数。各条轨迹等权，长轨迹不会在该指标中得到更多权重。

正式评测会校验实际选中 270 个任务及逐批真实 task ID；不符时直接报错，不能把一个意外缩小的任务集当作完整结果。若要所有原生 test variation，显式传 `--eval-protocol full`，该结果使用单独目录和 run 名。自动最终评测使用本次成功保存的**最终** checkpoint，不利用 test 选择模型，也不读取旧 tracker 猜测训练是否完成。

## 运行

```bash
# 更新服务器代码后，使用新实验名启动。需要已有模型、环境和 tasks.json。
EXPERIMENT_NAME=scienceworld_baseline_7b_seed0_swiftsage_n8_t50 \
  bash recipe/denoise_v2/run_scienceworld_grpo_train.sh

EXPERIMENT_NAME=scienceworld_denoise_7b_seed0_swiftsage_n8_t50 \
  bash recipe/denoise_v2/run_scienceworld_denoise_train.sh

# 单独正式评估基础模型，或已保存的 checkpoint。
bash recipe/denoise_v2/run_scienceworld_eval.sh --base-model
bash recipe/denoise_v2/run_scienceworld_eval.sh --checkpoint /path/to/global_step_500

# 所有原生 test variation；耗时会显著增加。
bash recipe/denoise_v2/run_scienceworld_eval.sh --checkpoint /path/to/global_step_500 --eval-protocol full

# GPU 短训练验通，不运行最终 test。
bash recipe/denoise_v2/run_scienceworld_grpo_train.sh --skip-final-eval \
  trainer.total_training_steps=2 trainer.total_epochs=2 trainer.test_freq=-1 trainer.save_freq=1
```

ScienceWorld 在作业环境有 `WANDB_API_KEY`、`WANDB_IDENTITY_TOKEN_FILE` 或当前 W&B 主机的 netrc 登录时默认 online；未发现这些凭据则打印提示并使用 offline。检查不访问网络、不打印密钥，也不验证服务端权限。已有显式 `WANDB_MODE` 始终优先；通过其他 SDK 设置保存凭据时，可显式设 `WANDB_MODE=online`。单独评估默认 console；如需 W&B，附加 `trainer.logger='[console,wandb]'`。自动最终评测继承训练的模式与 logger，使用独立 run，并移除训练的显式 W&B run ID，避免合并曲线。

如果日志报 `No API key configured`，在**实际执行作业的环境**运行 `wandb login`，或通过平台的秘密环境变量配置 `WANDB_API_KEY`。本机浏览器登录 W&B 不代表计算节点已登录；登录节点的 netrc 也需要能被作业容器读取。随后用 `WANDB_MODE=online` 运行原训练命令即可实时上传。若先离线训练，用 `WANDB_MODE=offline` 运行原命令；指标保存在控制台与本地 W&B 文件，网页不会实时更新，之后可登录并执行 `wandb sync /path/to/wandb/offline-run-...`。

## 指标与耗时

- 每次训练更新：`episode/reward/mean`、`episode/success_rate`、动作长度、损失及 timing；一次更新需要完成 rollout 和 PPO，不等于一个环境动作。
- dev/test：`val/<split>/score_macro` 为任务类型等权的 0–100 分数；`score` 为 episode 等权；`success_rate` 为完全成功率。保留旧的 0–1 `test_score` reward 指标。类型分数、无效动作率、选中任务覆盖率一并输出。
- 每 20 个动作轮打印剩余活跃轨迹数；每个评测 batch 输出完成数、当前均分、耗时、ETA，更新 W&B `eval_progress/<split>/*` summary。这些是进行中统计，不是最终 test 成绩。
- 本地保存 `progress.json`、最终 `<step>.summary.json` 与轨迹 `<step>.jsonl`。默认自动最终结果目录为 `recipe/denoise_v2/dumps/<experiment>/validation/final_swiftsage/test/`；最终报告的 `complete=true` 仅在覆盖率校验后写入。

prompt 改用原生有限动作模板与当前状态，不再展开上千个对象/动作组合。超过 token 预算先移除最早的历史，始终保留任务、当前状态和动作格式；这些必要内容本身超长则明确报错，避免悄悄从左侧截掉任务。

这些调整减少训练期间评测负担，不能保证一个 7B 多轮任务瞬间完成。仍使用原有 FSDP/vLLM 权重同步和批处理实现；CPU/JVM 数量、模型逐动作生成和任务长度仍影响耗时。已启动的旧服务器进程不会热更新，需同步代码后重新启动。旧采样数或环境/动作预算/计分/prompt 的 curriculum checkpoint 与新配置不兼容，应使用新实验名重训；仅做独立 checkpoint 评测不需要 curriculum 状态。
