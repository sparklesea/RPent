# RoboTwin BRACE、HSP-Plan 与 Stage-2 消融实验

本文说明三个 RoboTwin 实验入口的实现方式，供复现和二次开发使用：

- **BRACE baseline**：BRACE 风格的事件触发式 GPT 重规划；
- **HSP-Plan baseline**：首轮 GPT plan，中间按时间重规划；
- **Stage-2 random-search ablation**：协同调用规则的随机搜索消融。

三者统一使用：

| 角色 | 模型 | reasoning effort | memory |
|---|---|---|---|
| worker | `openai-chat:qwen3.8-27b` | `none` | 任务级 local memory |
| senior | `openai:gpt-6-astra` | `high` | 任务级 GPT correction memory（BRACE、HSP-Plan） |

两个 baseline 的评测规模为 50 个 RoboTwin 任务、entry index `2,3`、共 100 个
episode；两个 GPU 并行，每卡 5 个 worker，单 episode 上限 1200 s。Stage-2 消融
使用 entry index `1` 做规则搜索，因此只跑 50 个任务。

## 0. 统一模型与超参数

### 0.1 RoboTwin 固定口径

RoboTwin 实验统一采用下表配置。后续章节的命令和实现都以这组值为准：

| 项目 | 值 |
|---|---|
| worker | `openai-chat:qwen3.8-27b` |
| worker reasoning effort | `none` |
| senior | `openai:gpt-6-astra` |
| senior reasoning effort | `high` |
| episode timeout | `1200 s` |
| 周期 check-in | `8` 个 worker 请求 |
| BRACE 连续错误阈值 | `4` |
| BRACE 停滞窗口 | `4` 帧 |
| BRACE 停滞位移阈值 | `0.01 m` |
| BRACE 失败 finish veto | `1` 次 |
| HSP-Plan 重规划间隔 | `600 s`（1200 s 的一半，约 1 次） |
| Stage-2 搜索的 K | `8` 或 `4` |
| Stage-2 搜索的 N | `4`、`3` 或 `2` |

需要区分“周期 check-in”和“Stage-2 搜索的 K”：BRACE 的周期 check-in 固定为 8；
HSP-Plan 关闭周期 check-in，只保留定时 replan；Stage-2 消融把 K=8 作为默认候选，
同时把 K=4 纳入搜索空间。不要把 HSP-Plan 的 `time_replan` 统计成周期 check-in。

### 0.2 其他 benchmark 的建议起点

模型和 reasoning effort 建议保持不变：Qwen 3.8-27B 使用 `none`，GPT-6 Astra 使用
`high`。timeout 和 check-in 应按环境步长、单回合耗时和任务时长调整。以下是新
benchmark 的建议起点，不是固定契约：

| benchmark 类型 | episode timeout | 周期 check-in | 连续错误 N | HSP-Plan replan 间隔 |
|---|---:|---:|---:|---:|
| LIBERO / LIBERO-Pro | `600 s` | `8` | `4` | `300 s` |
| RoboCasa | `1200 s` | `8` | `4` | `600 s` |
| 其他长时序操作 benchmark | `1200 s` | `8` | `4` | `timeout / 2` |
| 回合很短、单步开销低的 benchmark | `300~600 s` | `6~8` | `3~4` | `timeout / 2` |

统一规则：

- HSP-Plan 的重规划间隔取 `episode_timeout / 2`；需要 2 次 replan 时取
  `episode_timeout / 3`。
- BRACE 的停滞窗口仍从 `4` 帧、`0.01 m` 起步；如果环境动作幅度天然很小，先提高
  位移阈值，再考虑增大窗口。
- 如果任务时长的 p95 已接近 timeout 的 80%，优先把 timeout 提高到 1800 s 或
  2400 s，而不是通过减小 check-in 来“催”模型。
- Stage-2 消融仍使用 `K ∈ {8, 4}`、`N ∈ {4, 3, 2}`；其他 benchmark 若要缩小搜索
  空间，至少保留一个较大的 K 和一个较小的 K，避免消融退化成单一配置。

> 本文中的 BRACE 是“复现其动态重规划控制器”的轻量 baseline，不是完整论文实现。
> 它没有实现 E-RECAP、retrieval、动态 token budget、latency SLO、cooldown 或
> commit window。HSP-Plan 也不是论文方法的逐行复刻，而是本文实验使用的
> GPT-plan / Qwen-execute 定时重规划基线。

---

## 1. 共同运行方式

### 1.1 依赖与环境变量

所有入口都通过 RPent 的 `--planner api` 运行。worker 与 senior 使用不同网关：

```bash
export INFINI_BASE_URL=https://cloud.infini-ai.com/maas
export INFINI_API_KEY=<Qwen API key>
export RPENT_BASE_URL=https://ai.soruxgpt.com
export RPENT_API_KEY=<GPT API key>
```

脚本会自动给 base URL 补 `/v1`。共同依赖还包括：

```bash
ASSETS=runs/robotwin-ecc-stage1/assets
VLA_MODEL=runs/robotwin-ecc-stage1/models/LingBot-VLA-RoboTwin-EEF-ckpt1500
QWEN_MEMORY=runs/robotwin-ecc-stage2-gpt-memory/memory
GPT_MEMORY=memory/robotwin
```

`QWEN_MEMORY` 是 worker 读取的任务级 memory；`GPT_MEMORY` 是 senior 使用的
RoboTwin GPT memory 源。评测脚本会把后者渲染成每个任务一个 Markdown 文件，并通过
`RPENT_CLOUD_CORRECTIONS` 只注入 senior prompt，不展示给 worker。

### 1.2 调度框架

两个 baseline 都复用 `rpent/planner/escalation.py` 中的 `EscalationModel`：

- 同一个 episode 共享同一份对话历史；
- 每个模型请求先经过 `_trigger()` 判断由 worker 还是 senior 回答；
- senior 的回答会被标记为 `[senior planner guidance]`；
- senior 只回复文本、没有工具调用时，harness 会把该文本交给 worker，再由 worker
  产生本轮工具调用；
- 每次请求写入 `escalation.jsonl`，包括 role、reason、model、token usage 和工具名。

因此两条 baseline 的差异只在“什么时候让 GPT 接管下一轮”，而不是两套独立执行器。

### 1.3 Qwen 缓存

Infini 上的 `qwen3.8-27b` 支持隐式前缀缓存，不需要显式 `cache_control`。同一个
episode 的 system prompt、工具 schema 和前序历史保持稳定时，后续请求会命中缓存；
`[usage] cache_read=...` 会显示真实命中量。BRACE、HSP-Plan 和消融脚本无需额外改动。

---

## 2. BRACE baseline

### 2.1 策略定义

BRACE baseline 的 policy 名为：

```text
gpt_plan_qwen_execute_brace_event_replan
```

Qwen 执行大多数回合。GPT 负责：

1. 初始 plan；
2. 连续工具错误后的重规划；
3. 末端执行器停滞后的重规划；
4. 周期性进度检查；
5. worker 提出失败 `finish` 时的一次 veto。

入口文件：

- `scripts/ecc_robotwin/run_brace_eval.py`
- `scripts/ecc_robotwin/run_brace_episode.sh`
- `scripts/ecc_robotwin/rpent_brace.py`

### 2.2 初始 plan

初始时 `planned=False`。当工具结果中同时出现：

```text
read_text_file
view_env_state
```

说明 worker 已经读完 memory 和初始场景，下一次请求交给 GPT 做初始 plan。若到第 6
个请求仍未满足条件，也会强制触发 `plan`，避免 worker 卡在准备阶段。

### 2.3 事件触发条件

GPT 完成初始 plan 后，BRACE 只检查“senior 上次审阅之后新产生的工具结果”，避免同一
批证据每轮重复触发。

| 触发原因 | 条件 | 默认值 |
|---|---|---|
| `tool_errors` | 最新 `N` 个 fresh 工具结果都包含 `error` | `N=4` |
| `eef_stall` | 最近 `W` 个有效末端位姿窗口内，最大位移小于 `S` 米 | `W=4`, `S=0.01` |
| `periodic_checkin` | 距离上次 senior 已过去 `K` 个 worker 请求 | `K=8` |
| `giveup` | worker 调用 `finish` 且 status 不是 `success` | 最多 veto 1 次 |

停滞检测读取 RoboTwin 的 `left_eef_pose` 和 `right_eef_pose`。对每个 arm，以窗口
首帧为基准，计算后续帧与首帧的最大欧氏距离，再取所有可用 arm 的最大值；只有该
最大值小于阈值时才认为末端执行器停滞。窗口不足 4 帧时不触发。

BRACE 的 `max_share` 固定为 `1.0`，即不限制 senior 占总请求的比例；是否接管完全由
上述事件决定。`max_vetoes=1`，所以 worker 提出失败 `finish` 后，GPT 最多接管一次。

### 2.4 推荐运行命令

```bash
PYTHON_BIN=.runtime/venv/bin/python

$PYTHON_BIN scripts/ecc_robotwin/run_brace_eval.py \
  runs/robotwin-brace-event-eval-qwen38 \
  --qwen-memory-dir "$QWEN_MEMORY" \
  --gpt-memory-source "$GPT_MEMORY" \
  --assets-path "$ASSETS" \
  --model-path "$VLA_MODEL" \
  --gpus 0,1 \
  --workers-per-gpu 5 \
  --entry-indexes 2,3 \
  --timeout-s 1200 \
  --cleanup-grace-s 60 \
  --checkin 8 \
  --errors 4 \
  --stall-window 4 \
  --stall-span-m 0.01 \
  --max-vetoes 1
```

`--timeout-s 1200` 是外层 episode watchdog；planner/tool timeout 为
`1200 - cleanup_grace_s`，默认即 1140 s。外层超时后会终止整个进程组，避免子进程残留。

### 2.5 结果判定

BRACE 的 `result.json` 同时记录两种成功：

- `solved`：环境报告 `eval_success=true`，与其它评测口径一致；
- `clean_success`：`solved`、进程正常退出、且 `finish(status="success")` 三者同时成立。

主表若比较环境成功率，使用 `solved`；若比较“干净完成率”，使用 `clean_success`。
`summary.json` 还会输出成功 episode 的平均 GPT 调用数、平均 turn 数和平均耗时。

---

## 3. HSP-Plan baseline

### 3.1 策略定义

HSP-Plan baseline 的 policy 名为：

```text
gpt_plan_qwen_execute_timed_replan
```

执行过程：

1. 先由 GPT 做一次初始 plan；
2. 其余回合由 Qwen 执行；
3. episode 仍在运行时，每隔 `replan_interval_s` 让 GPT 再做一次 plan；
4. 不启用工具错误、末端停滞、周期 check-in 或失败 finish 的 veto。

入口文件：

- `scripts/ecc_robotwin/run_plan_timed_eval.py`
- `scripts/ecc_robotwin/run_plan_timed_episode.sh`
- `scripts/ecc_robotwin/rpent_plan_timed.py`

### 3.2 初始 plan

初始 plan 与 BRACE 共用 `EscalationModel` 的触发条件：memory 和场景都读取完成后，
或者请求数达到 6，GPT 接管第一轮。

### 3.3 定时重规划

`PlanTimedModel` 在构造时记录 `start_time`，下一次重规划时间为：

```text
next_time_replan = start_time + replan_interval_s
```

当模型请求发生时，如果 `time.monotonic() >= next_time_replan`，reason 记为
`time_replan`，GPT 接管本轮；触发后：

```text
next_time_replan = now + replan_interval_s
```

因此 1200 s episode 的常用配置是：

- `--replan-interval-s 600`：约在第 600 s 重规划 1 次；
- `--replan-interval-s 400`：约在第 400、800 s 重规划 2 次。

重规划只会在一次新的模型请求开始时检查，所以时间是“至少经过该间隔后的下一次请求”，
不是精确的 wall-clock 定时器。

HSP-Plan 通过以下设置关闭通用升级路径：

- `checkin=10^9`；
- `errors=10^9`；
- `max_vetoes=0`；
- `max_share=1.0`。

### 3.4 推荐运行命令

```bash
$PYTHON_BIN scripts/ecc_robotwin/run_plan_timed_eval.py \
  runs/robotwin-plan-timed-eval-qwen38 \
  --qwen-memory-dir "$QWEN_MEMORY" \
  --gpt-memory-source "$GPT_MEMORY" \
  --assets-path "$ASSETS" \
  --model-path "$VLA_MODEL" \
  --gpus 0,1 \
  --workers-per-gpu 5 \
  --entry-indexes 2,3 \
  --timeout-s 1200 \
  --subprocess-timeout-s 1200 \
  --replan-interval-s 600
```

### 3.5 结果判定

每个 episode 的 `result.json` 记录 `senior_reason_counts`；HSP-Plan 的主信号是：

```text
"time_replan": 1
```

`summary.json` 汇总成功数、GPT 调用数和结束时间。若需要区分“首次 plan”和“定时
replan”，直接统计 `escalation.jsonl` 中 `role=senior` 且 `reason=time_replan` 的行。

---

## 4. Stage-2 random-search ablation

### 4.1 实验目标

正式 stage-2 搜索使用一组人工排序的 6 个协同规则，逐个尝试直到任务成功。消融实验
保持同样的“最多 6 次配置尝试”预算，但把前 5 个配置改为随机采样，用来判断规则搜索
收益是否依赖人工排序。

入口文件：

- `scripts/ecc_robotwin/run_ecc_stage2_random_qwen38.py`
- `scripts/ecc_robotwin/run_ecc_stage2_qwen38_episode.sh`

### 4.2 配置空间

完整配置空间是 12 个笛卡尔积：

```text
plan ∈ {qwen_plan, gpt_plan}
K    ∈ {8, 4}
N    ∈ {4, 3, 2}
```

其中：

- `plan`：是否让 GPT 做初始 plan；
- `K`：`checkin_K`，每 K 个 worker 请求允许一次 GPT 周期检查；
- `N`：`errors_N`，连续 N 个工具错误允许一次 GPT 接管。

fallback 固定为：

```text
gptplan_K4_N2
```

### 4.3 每个任务的搜索顺序

对每个任务：

1. 用任务级确定性 RNG 从 11 个非 fallback 配置中无放回抽取 5 个；
2. 把 `gptplan_K4_N2` 追加为第 6 个配置；
3. 按顺序执行配置，遇到第一个成功即停止；
4. 如果 6 个配置都失败，记录 fallback 结果。

任务级 RNG 的 seed 文本为：

```text
ecc-stage2-random:{random_seed}:{task_name}:{seed}
```

因此相同 `random_seed` 下，每个任务的随机顺序可复现，而且任务之间相互独立。默认
`random_seed=0`。

### 4.4 运行配置

| 项目 | 设置 |
|---|---|
| worker | `openai-chat:qwen3.8-27b`，reasoning `none` |
| senior | `openai:gpt-6-astra`，reasoning `high` |
| entry index | `1` |
| 任务数 | 50 |
| worker memory | `runs/robotwin-ecc-stage2-gpt-memory/memory` |
| 每任务最多尝试 | 6 个配置 |
| 单配置 timeout | 1200 s |
| 并行 | 2 GPU × 5 worker |

stage-2 搜索中没有单独的 GPT correction memory 文件；senior 直接读取共享对话中的
任务 memory 和工具结果。

### 4.5 推荐运行命令

```bash
$PYTHON_BIN scripts/ecc_robotwin/run_ecc_stage2_random_qwen38.py \
  runs/robotwin-ecc-stage2-random-k8-k4-qwen38 \
  runs/robotwin-ecc-stage2-random-k8-k4-qwen38/rules \
  "$QWEN_MEMORY" \
  --assets-path "$ASSETS" \
  --model-path "$VLA_MODEL" \
  --gpus 0,1 \
  --workers-per-gpu 5 \
  --entry-index 1 \
  --random-seed 0 \
  --timeout-s 1200 \
  --subprocess-timeout-s 1200 \
  --rpent .runtime/venv/bin/rpent
```

### 4.6 产物

```text
runs/robotwin-ecc-stage2-random-k8-k4-qwen38/
├── batch.json
├── console.log
├── status.json
├── summary.json
├── runner.log
├── rules/<task>.json
└── <task>_s<seed>/
    └── rs0_s01_<config>/
        ├── result.json
        ├── run.log
        └── stdout.log
```

`rules/<task>.json` 是单个任务的搜索记录，关键字段：

- `sampled_configs`：随机抽取的 5 个配置；
- `search_order`：5 个随机配置加最后的 fallback；
- `attempted_configs`：实际执行过的配置；
- `solved_in_search`：是否至少有一个配置成功（包括 fallback）；
- `success_config` / `success_position`：成功配置及其顺序；
- `fallback_attempted` / `fallback_failed`：是否用到 fallback 及 fallback 是否失败；
- `history`：每个配置的 success、GPT 调用、token、耗时和 timeout 情况；
- `gpt_cost_usd_in_search`：该任务搜索期间的 GPT 成本。

`summary.json` 中的 `solved_in_search` 表示“至少有一个配置成功”，不是最终评测集的
成功率；`average_configs_per_task` 才是搜索成本的核心指标。

### 4.7 恢复语义

脚本按 `rules/<task>.json` 恢复：

- 已完成且 `algorithm`、`random_seed` 匹配的任务直接跳过；
- 未完成的任务从其保存的随机顺序重新开始；
- 已存在的 `result.json` 会复用，不重复执行该配置。

恢复键不包含模型名。**不要**把 qwen3.8 脚本指向 qwen3.6 的旧 `rules` 目录，否则会
把旧模型结果当成新实验的已完成任务。BRACE 和 HSP-Plan 的 `result.json` 虽然记录
`worker_model`，但恢复时不会校验该字段，同理应使用新的输出目录。

---

## 5. 选择建议

- 想看“事件驱动重规划”的效果：用 BRACE baseline，重点比较 `solved`、
  `clean_success`、`senior_reason_counts` 和 GPT 调用数。
- 想看“固定时间预算下 GPT 介入几次”的效果：用 HSP-Plan，重点比较
  `time_replan` 次数、GPT 调用数和成功率。
- 想看 stage-2 规则搜索是否依赖人工排序：用 random-search ablation，重点比较
  `average_configs_per_task`、`solved_in_search`、fallback 使用率和搜索成本。

三套实验都应使用独立输出目录，避免不同模型或不同搜索种子的结果互相复用。
