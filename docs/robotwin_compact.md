# RoboTwin Compact 工具接口精简

本文说明 `robotwin_compact` 的实现、调用链、评测方法和实测结果。目标不是介绍一个
“新模型”，而是解释如何在不改变仿真环境、VLA 和动作语义的前提下，减少模型与工具
之间协议层的冗余 token。

一句话概括：

> **模型–工具接口精简（Tool Interface Compaction）**
>
> 把模型不需要知道、不需要生成、或者运行时本来就能确定的字段，从工具 schema、
> 工具参数、工具结果和 prompt 中移除；保留完整的观测、动作和任务执行能力。

---

## 0. 先看结论

`robotwin_compact` 做了四类修改：

1. **memory 读取改为短引用**：模型生成 `read_memory({"ref": "task_info"})`，不再
   生成包含绝对路径的 `read_text_file({"path": "..."})`。
2. **VLA 调用参数缩减**：模型只生成 `lingbot_act({"chunks": 2})`；任务语言由当前
   environment observation 提供，`use_length` 使用固定值 50。
3. **工具结果回显缩减**：`lingbot_act` 返回时不再把 task language、
   `agent_prompt_ignored` 和 `ignored_agent_prompt` 回显给模型。
4. **prompt 同步改写**：把原来的路径式 read order、完整 VLA 参数说明替换为
   `read_memory` 和 compact `lingbot_act` 的规则，避免 prompt 和 schema 互相矛盾。

没有修改的部分同样重要：

- 不压缩图像观测、depth、world map、机器人状态或 memory 正文。
- 不改变 LingBot-VLA 模型、动作频率、action horizon 或控制语义。
- 不改变 RoboTwin 的成功判定、任务定义、seed 或 checkpoint。
- 不改变 Qwen + GPT 的 K/N 升级规则；compact 只作用于 Qwen worker 的工具接口。

主实验使用 50 个 RoboTwin 任务、entry `2,3`、每个 entry 一个 seed，共 100 个
episode。与同一 K8/N4 对照批次相比：

| 指标 | K8/K4 对照 | Compact | 变化 |
|---|---:|---:|---:|
| 成功率 | 57/100 | **56/100** | -1 episode |
| 成功 episode 平均 GPT 调用 | 0.684 | **0.482** | -29.5% |
| 成功 episode 平均耗时 | 408.3 s | **382.2 s** | -6.4% |
| 1200 s subprocess timeout | 41 | **30** | -11 |
| 双方都成功 episode 的输出 token / episode | 1337.5 | **974.7** | -27.1% |
| 双方都成功 episode 的输出 token / turn | 176.1 | **125.2** | -28.9% |
| 双方都成功 episode 的输入 token / episode | 132920.5 | **118575.9** | -10.8% |

主结果按 **56/100** 记录。compact 不是靠牺牲动作能力换 token；它的收益主要来自
协议层，而不是 VLA 策略本身。

---

## 1. 问题定义

### 1.1 三个概念

#### Tool interface

Tool interface 是模型与运行时之间的完整协议，包括：

- 工具名；
- 工具描述；
- JSON input schema；
- 模型生成的 tool_use 参数；
- 工具执行后的 result；
- result 被追加到对话历史后占用的上下文。

只要一次工具调用进入对话历史，它的参数和结果就会在后续每个 API request 中重复
出现。因此缩短一次调用，不只是省这一次的输出 token，也会影响后续所有 request 的
输入 token。

#### Tool argument token

模型生成工具调用时，工具名和 JSON 参数都由模型 token by token 生成。例如：

```json
{
  "chunks": 2,
  "use_length": 50,
  "prompt": "Pick the medium green bottle upright from the table"
}
```

其中 `chunks` 是有决策意义的；`use_length` 在当前模型上是固定常量；`prompt` 应该
等于环境已经提供的 task language。后两个字段属于“模型被要求重复运行时事实”。

#### Protocol redundancy

本文把冗余分为三类：

| 类型 | 例子 | 是否影响物理执行 |
|---|---|---|
| 路径冗余 | memory 绝对路径 | 不影响，文件内容才影响 |
| 运行时事实冗余 | `use_length=50`、完整 task language | 不影响，运行时本来就知道 |
| 回显冗余 | result 中的 `prompt`、ignore flags | 不影响，模型刚生成过或根本不需要 |

Compact 只移除这三类冗余，不移除动作信息。

### 1.2 原始评测接口

原始 RoboTwin 评测中，模型一般会这样读 memory：

```json
read_text_file({
  "path": "/inspire/.../runs/robotwin-ecc-stage2-gpt-memory/memory/task-specific/robotwin_adjust_bottle_demo_randomized_s0.json"
})
```

工具返回还会带回：

```json
{
  "path": "/inspire/.../robotwin_adjust_bottle_demo_randomized_s0.json",
  "size": 2042,
  "content": "..."
}
```

这里真正影响决策的是 `content`。`path` 和 `size` 对模型没有语义价值，却会随着每次
读取进入历史上下文；绝对路径越长，重复成本越高。

原始 `lingbot_act` 的典型调用是：

```json
{
  "chunks": 2,
  "use_length": 50,
  "prompt": "Pick the medium green bottle upright from the table"
}
```

底层 `RoboTwinPrimitives.lingbot_act` 并不使用模型传入的 `prompt` 作为策略输入；它
每一轮从当前 observation 读取 `task_language`。模型传入的 prompt 只会被记录到
`ignored_agent_prompt`，并通过 `agent_prompt_ignored` 标记“被忽略”。`use_length`
也必须是 50，否则 primitive 直接报错。因此这两个参数是“被迫生成但不会被采用”的
字段。

### 1.3 设计目标

Compact 的目标按优先级排列：

1. **不损害任务成功率**：动作能力和 memory 正文保持不变。
2. **减少模型输出长度**：尤其是每次 `lingbot_act` 的重复 task language。
3. **减少历史上下文膨胀**：让短参数和短结果在后续 request 中重复得更少。
4. **减少无效参数错误**：模型不再有机会传错 `use_length` 或旧 task language。
5. **保持可审计性**：每个 ref 仍然能解析到明确文件，每个动作仍可从日志回放。

### 1.4 非目标

以下事情不在 compact 范围内：

- 不压缩 memory 正文，也不做摘要、检索或 RAG。
- 不减少图像数量，不降低分辨率。
- 不压缩 world map 的坐标或 observation 的机器人状态。
- 不改变 `chunks` 与 50-step action chunk 的对应关系。
- 不压缩 `finish` 的 summary，也不改变成功判定。
- 不把 Qwen 或 GPT 变成“无工具调用”的纯文本模型。
- 不让模型直接执行 shell、Python 或任意文件系统访问。

---

## 2. 接口差异总览

### 2.1 原始接口与 compact 接口

| 操作 | 原始接口 | Compact 接口 | 运行时补充 |
|---|---|---|---|
| 读取 memory | `read_text_file({"path": "..."})` | `read_memory({"ref": "task_info"})` | ref 到白名单文件的映射 |
| 读取 guide | 完整 repo path | `read_memory({"ref": "guide"})` | 返回前修补旧接口说明 |
| VLA 动作 | `{"chunks", "use_length", "prompt"}` | `{"chunks"}` | task language + `use_length=50` |
| VLA 结果 | 含 prompt 和 ignore flags | 只保留执行状态 | wrapper 删除回显字段 |
| 文件枚举 | `list_dir` | 不暴露 | 固定 ref 列表替代动态枚举 |

### 2.2 最终工具表

`CompactRoboTwinToolkit` 先继承基础 `Toolkit` 的 common tools，再调用基础
`RoboTwinToolkit._register_robotwin_tools()`，最后做 compact 覆盖。最终正常情况下
暴露的工具是：

| 类别 | 工具 | Compact 是否保留 | 说明 |
|---|---|---|---|
| 文件 | `write_text_file` | 保留 | 仍受 `MemoryManager` 的 memory access 约束 |
| 文件 | `read_text_file` | 删除 | 用 `read_memory` 替代 |
| 文件 | `list_dir` | 删除 | 不允许模型动态发现文件 |
| memory | `read_memory` | 新增 | 白名单短引用 |
| 观测 | `view_env_state` | 保留 | 完整 observation |
| 观测 | `sample_world_xyz` | 保留 | 坐标采样 |
| 观测 | `query_world_map` | 保留 | world map 区域查询 |
| 观测 | `render` | 保留 | 当前实现为轻量 no-op |
| 控制 | `lingbot_act` | 覆盖为 compact schema | 只接收 `chunks` |
| 控制 | `move_to` | 保留 | 不压缩 |
| 控制 | `rotate_wrist` | 保留 | 不压缩 |
| 控制 | `set_gripper` | 保留 | 不压缩 |
| 控制 | `release` | 保留 | 不压缩 |
| 终止 | `finish` | 保留 | 仍要求 status + summary |
| 探索 | `reset` | 不暴露 | evaluation only |

此外，API planner 会另外注入 `read_image` 工具。它不是 RoboTwin toolkit 的 common
tool，因此不在 `_register_robotwin_tools()` 的删除范围内；compact 仍然允许用
`read_image` 查看已经持久化的图像 artifact。不要把“删除 read_text_file/list_dir”
误解成“删除所有文件/图像入口”。

### 2.3 为什么保留 `write_text_file`

基础 `Toolkit` 注册了 `write_text_file`，compact 没有显式 pop 它。它仍然受
`MemoryManager(memory_access="read_only")` 的检查：向当前 memory 的已发布区域写入
会被拒绝，向其他 robot 的 memory 写入也会被拒绝。保留它是为了避免在 toolkit 层
引入不必要的行为变化；实际评测中模型不需要它，prompt 也不会要求它写文件。

---

## 3. 代码结构

实现全部位于新增的 `robots/robotwin_compact/`，不修改原始 `robots/robotwin/` 的
执行逻辑。

| 文件 | 职责 |
|---|---|
| `robots/robotwin_compact/__init__.py` | 导出 RobotSpec 和 toolkit 工厂 |
| `robots/robotwin_compact/robot_spec.py` | 包装原 spec、禁用 exploration、构造 toolkit |
| `robots/robotwin_compact/toolkit.py` | 实现 `read_memory` 和 compact `lingbot_act` |
| `robots/robotwin_compact/prompts.py` | 替换与接口有关的 prompt 段落 |
| `scripts/ecc_robotwin/run_compact_eval.py` | 复用批量评测并替换 episode runner |
| `scripts/ecc_robotwin/run_compact_eval_episode.sh` | 单 episode 启动脚本 |

### 3.1 RobotSpec 包装

`robot_spec.py` 没有复制一套 Env/VLA/primitives，而是使用 `dataclasses.replace`：

```python
return replace(
    base.get_robot_spec(),
    name="robotwin_compact",
    prompts=PromptBundle(system=..., user=...),
    supports_exploration=False,
)
```

这保证底层仍然是同一套：

- RoboTwin EnvServer；
- LingBot-VLA server；
- EEF action layout；
- `RoboTwinPrimitives`；
- `EnvState` artifact 和 step 记录。

变化只在名字、prompt bundle、toolkit 和 exploration 开关。

### 3.2 Toolkit 初始化时机

初始化顺序可以概括为：

```text
CompactRoboTwinToolkit.__init__
  -> 记录 reference_tag
  -> RoboTwinToolkit.__init__
       -> Toolkit.__init__
            -> _register_common_tools()
                 read_text_file / write_text_file / list_dir / finish
       -> 构造 RoboTwinPrimitives
       -> evaluation: 读取 env.last_reset_info，不额外 reset
       -> _register_robotwin_tools()
            -> super()._register_robotwin_tools()
                 pop finish
                 add observation + control + finish
            -> pop read_text_file
            -> pop list_dir
            -> overwrite lingbot_act
            -> build memory refs
            -> add read_memory
       -> get_env_state(step=0)
```

注意：`reference_tag` 必须在 `super().__init__()` 之前保存，因为
`_register_robotwin_tools()` 在父类构造期间就会被调用，而 memory refs 需要这个值。

### 3.3 Mode 限制

`get_toolkit()` 在 `mode != "evaluation"` 时抛出 `ValueError`。因此 compact 不能用于：

- 探索；
- `reset` 后多次尝试；
- recipe 写入；
- memory 合并；
- exploration session 的 handoff。

这不是一个“通用 robot 压缩插件”，而是 RoboTwin evaluation 的变体。

---

## 4. Memory 短引用协议

### 4.1 Ref 从哪里来

`_memory_refs()` 以 `MemoryManager.root` 为根生成：

| Ref | 目标文件 | 说明 |
|---|---|---|
| `guide` | `robots/robotwin/guides/GUIDE_RPENT.md` | 操作指南 |
| `task_info` | `<memory>/task-specific/<reference_tag>.json` | 当前任务语义 recipe |
| `task_recipe` | `<memory>/task-specific/<reference_tag>_recipe.jsonl` | 历史动作序列 |
| `memory_index` | `<memory>/MEMORY.md` | global memory 索引 |
| global stem | `<memory>/global/*.md` | 例如 `grasp_and_hold` |

`reference_tag` 由 `RunConfig.prompt_vars` 提供，取决于 memory profile：

```text
local profile: robotwin_<task>_<task_config>_s0
hf profile:    <task>_s0
```

本批 compact 单 episode 使用 `--memory-profile local` 和
`--memory-dir <QWEN_MEMORY_DIR>`，所以默认是：

```text
robotwin_adjust_bottle_demo_randomized_s0
```

### 4.2 为什么用 ref，而不是路径

路径有三个问题：

1. **长度**：一个 memory 绝对路径通常 100–220 个字符。
2. **重复**：同一个路径在多次请求的历史中反复出现。
3. **语义噪声**：模型只需要“这是 task_info”，不需要知道 repo 在哪个挂载盘。

Ref 把协议从：

```text
path -> open -> file
```

变成：

```text
ref -> allowlisted resolver -> file
```

这既缩短了参数，也把文件访问边界从“文件系统”收紧到“当前任务允许的文档”。

### 4.3 Schema 和枚举

`read_memory` 的 schema 是：

```json
{
  "type": "object",
  "properties": {
    "ref": {
      "type": "string",
      "enum": ["guide", "task_info", "task_recipe", "..."]
    }
  },
  "required": ["ref"]
}
```

枚举值是 `sorted(refs)`。这样做有两个效果：

- 模型在结构上不能传任意路径；
- 模型可以直接看到当前可用的 ref，不需要 `list_dir`。

代价是：global 文件很多时，枚举本身会变长；ref 表是在 toolkit 初始化时生成并写入
schema 的，新增 global 文件后需要新的 run 才能稳定出现在 schema 里。虽然 handler
每次调用会重新计算 refs，但模型侧看到的 enum 仍然是初始化时的那一份。

### 4.4 解析算法

`_read_memory()` 做的是严格白名单解析，不是路径拼接：

```python
normalized = ref.strip()
if normalized.startswith("global/"):
    normalized = normalized[len("global/"):]
if normalized.endswith(".md"):
    normalized = normalized[:-3]

path = refs.get(normalized)
if path is None:
    return {"error": "unknown memory ref", "available_refs": ...}
```

因此下面两种写法等价：

```json
{"ref": "grasp_and_hold"}
{"ref": "global/grasp_and_hold.md"}
```

但以下都不会被接受：

- `global/../task-specific/...`：不在 refs 表里；
- 绝对路径；
- 大小写不同的 stem；
- 不带 `.md` 的未知文件名；
- task-specific 的相对路径。

即使 planner 或 API 层没有遵守 enum，最终也会在 `refs.get()` 处失败，而不是打开
任意文件。

### 4.5 返回值和错误

成功时返回：

```json
{
  "ref": "task_info",
  "content": "..."
}
```

不返回路径、文件大小、目录列表或其他文件系统元数据。

未知 ref 返回：

```json
{
  "error": "unknown memory ref: ...",
  "available_refs": ["guide", "task_info", "..."]
}
```

读取失败返回：

```json
{
  "error": "memory ref unavailable: task_info",
  "detail": "..."
}
```

`detail` 会保留底层 `OSError` 的信息，便于定位文件不存在、权限问题或路径是目录。
这不是把错误吞掉；错误仍然以结构化形式暴露给模型。

### 4.6 截断

正文超过 40,000 个字符时：

```python
content = content[:40000] + "\n\n[TRUNCATED at 40000 chars]"
```

注意这里是字符数，不是字节数，也不保证 UTF-8 编码后的字节数不变。对当前
RoboTwin memory 文件通常不会触发；如果未来 memory 变长，需要意识到截断发生在工具
层，而不是 memory 写入层。

### 4.7 Guide 修补

基础 guide 仍然描述旧接口。为了避免“prompt 让模型用 read_memory，但 guide 让模型
用 read_text_file”的冲突，`read_memory(ref="guide")` 在返回前做三组精确字符串替换：

1. `list_dir` / `read_text_file` -> `read_memory`；
2. “每次 lingbot_act 使用完整 task language 和 use_length=50” -> “只传 chunks，
   runtime 自动补全”；
3. “primitive 之后带完整 task language 继续” -> “primitive 之后直接调用
   lingbot_act，runtime 使用当前 task language”。

这是当前实现最“脆”的地方：替换基于精确文案匹配。如果 guide 以后改词，替换可能
失效，但不会改变工具执行；只是模型可能看到旧的操作说明。修改 guide 时需要同步
检查 `_GUIDE_REPLACEMENTS`。

### 4.8 `task_recipe` 中的旧字段

一个重要细节：compact 只修补 `guide`，不会改写历史 `task_recipe` JSONL。历史
recipe 仍可能长这样：

```json
{"action":"lingbot_act","chunks":2,"use_length":50,"prompt":"..."}
```

模型会看到这些旧字段，但 compact `lingbot_act` 的 schema 只接受 `chunks`。这些旧
字段是“历史证据”，不是当前工具协议；模型不能直接把整行 JSON 复制成新的 tool call。
这属于保留的残余冗余，也是为什么 compact 不能把输入 token 降到接近零的原因之一。

### 4.9 GPT memory 不走这个协议

`read_memory` 只服务 Qwen worker 的本地 memory。GPT senior 的 per-task memory 由
`run_collab_eval.py` 的 `render_gpt_memory()` 从官方 GPT corpus 渲染到
`<run_dir>/gpt-memory/<task>.md`，再通过 `RPENT_CLOUD_CORRECTIONS` 注入 escalation
hook。它不使用 `read_memory`，也不受 compact ref 表影响。

---

## 5. Compact `lingbot_act`

### 5.1 原始 schema

基础 `robots/robotwin/tools.py` 的定义包含：

```json
{
  "type": "object",
  "properties": {
    "chunks": {"type": "integer", "minimum": 1, "default": 4},
    "use_length": {"type": "integer", "const": 50, "default": 50},
    "prompt": {"type": ["string", "null"]}
  }
}
```

基础定义甚至没有把任何字段设为 required，模型可以省略参数，也可以生成完整任务
文本。实际历史运行中，模型通常三个字段都生成。

### 5.2 Compact schema

`_COMPACT_LINGBOT_SPEC` 把协议缩为：

```json
{
  "type": "object",
  "properties": {
    "chunks": {
      "type": "integer",
      "minimum": 1,
      "maximum": 3
    }
  },
  "required": ["chunks"]
}
```

模型调用变成：

```json
{"chunks": 2}
```

`chunks` 仍然是模型真正需要决策的唯一变量：

- `1`：接触、接近成功、状态不稳定、需要小修正；
- `2`：普通稳定推进；
- `3`：recipe 支持、已经正确运动、需要连续性的阶段。

这里 `chunks` 不是“动作条数”，而是“VLA chunk 个数”。每个 chunk 最多执行
`use_length=50` 个 native actions，所以 `chunks=2` 的理论上限是 100 个动作。

### 5.3 执行链

```text
model tool call: lingbot_act({"chunks": 2})
  -> CompactRoboTwinToolkit._compact_lingbot_act(chunks=2)
  -> RoboTwinToolkit._step("lingbot_act", chunks=2)
  -> RoboTwinPrimitives.lingbot_act(chunks=2)
       -> for each chunk:
            observation = _build_lingbot_observation()
            native_prompt = observation["task_language"]
            actions = model.infer(observation)[:50]
            env.chunk_step(actions, action_type="ee")
  -> result cleanup
  -> planner tool result
```

`_compact_lingbot_act()` 本身只做两件事：

1. 调用 `_step("lingbot_act", chunks=chunks)`；
2. 删除 result 中的三个回显字段。

它不重新实现 VLA 推理，也不复制动作执行逻辑。

### 5.4 Task language 从哪里来

`_build_lingbot_observation()` 每一轮从环境读取：

```python
{
    "views": {...},
    "robot_state": env.last_info["robot_state"],
    "task_language": env.get_task_language(),
}
```

因此 compact 没有“让 runtime 使用模型传来的任务文本”；准确说法是：

> runtime 使用当前 environment 的 task language，模型不再被要求重复这段文本。

这比“把模型传来的 prompt 当作策略输入”更安全，因为：

- 不会使用过期的历史 task language；
- 不会因为模型改写任务文本而偏离 benchmark；
- 不需要维护 prompt 与 env 的一致性检查。

### 5.5 Chunk 循环和提前结束

`lingbot_act()` 的循环在每个 chunk 前检查：

```python
status = self.env.last_info["episode_status"]
if status.get("eval_success") is True:
    break
if status.get("step_lim") is not None and status.get("take_action_cnt", 0) >= step_lim:
    break
```

然后在每个 chunk 中执行：

```python
observation = self._build_lingbot_observation()
actions = self.model.infer(observation)[:MODEL_SPEC.use_length]
self.env.chunk_step(actions, action_type="ee")
```

因此请求的 `chunks=3` 不保证执行 150 个动作。可能：

- 第一个 chunk 就 native success，实际执行 50 个动作；
- 预算耗尽，执行 0 个或少于 50 个；
- 每个 chunk 实际执行数由 `info["executed_actions"]` 决定；
- `requested_steps = chunks * 50`，`executed_steps` 是实际值。

### 5.6 Completion 语义

`_completion()` 返回：

```json
{
  "completed": true,
  "requested_steps": 100,
  "executed_steps": 100,
  "stop_reason": "completed"
}
```

可能的 `stop_reason`：

| 值 | 含义 |
|---|---|
| `native_success` | 环境报告 `eval_success=true` |
| `budget_exhausted` | native step 预算耗尽 |
| `completed` | 请求的所有动作执行完，但未成功 |
| `runtime_failure` | 执行数少于请求数，且不是上述原因 |

注意：`completed=true` 只表示“请求的动作执行完了”，不表示任务成功。任务成功必须看
`eval_success` 或最终的原生状态。

### 5.7 结果字段清理

基础 primitive 返回结果中有：

```json
{
  "prompt": "...",
  "agent_prompt_ignored": false,
  "ignored_agent_prompt": null
}
```

Compact wrapper 在返回 planner 之前 pop 掉这三个 key。保留的是：

- `completed`
- `requested_steps`
- `executed_steps`
- `stop_reason`
- `success`
- `episode_status`

以及 `Toolkit` 随后附加的 step 记录和 observation artifact。清理是 in-place 对结果
dict 做的，不会改变 primitive 内部状态。

### 5.8 为什么这个改动是安全的

三个字段的行为已经由底层保证：

1. `prompt`：primitive 用 `observation["task_language"]`，不用参数；
2. `agent_prompt_ignored`：只有模型传了 prompt 才为 true，compact 下永远不传；
3. `ignored_agent_prompt`：compact 下永远是 None。

所以 compact 不是“隐藏一个重要参数”，而是删除一个已经被底层明确忽略的参数。

### 5.9 可能的风险

- 如果未来某个 VLA 版本真的使用模型传入的 prompt，compact 的假设就不再成立。
- 如果未来 `use_length` 不再是固定 50，compact schema 需要重新设计。
- schema 的 `maximum: 3` 是协议约束；primitive 本身只检查 `chunks >= 1`，并没有
  自己的上限。因此不要绕过 toolkit 直接调用 primitive 并假设它一定会限制到 3。
- 如果调用方不经过 JSON schema 校验，仍可能传入 `chunks=4`；实际行为取决于
  `RPENT_COERCE_TOOL_ARGS` 和 planner 的校验路径。

---

## 6. Prompt 改动

### 6.1 为什么 schema 变了，prompt 也必须变

模型行为由两部分共同约束：

- system prompt 里的策略说明；
- tool schema 里的可用参数。

如果只改 schema 而保留旧 prompt，模型会看到“每次 lingbot_act 要传完整 task
language”的文字，但 schema 不允许该字段，容易产生：

- 无效参数；
- 反复尝试旧格式；
- 误以为工具调用失败；
- 额外的重试和 token 消耗。

因此 compact 同步替换 prompt 中与接口相关的段落。

### 6.2 替换了哪些 key

`prompts.py` 复制基础 evaluation prompt，然后替换：

| Prompt key | 原始内容 | Compact 内容 |
|---|---|---|
| `READ ORDER` | 用 `{{memory_dir}}` 和 `{{reference_tag}}` 拼路径 | 用 `read_memory(ref=...)` |
| `VLA AND PRIMITIVE CONTROL` | 每次 VLA 传完整 task language + `use_length=50` | 只传 `chunks`，runtime 补全 |
| `RUNTIME` | `{{memory_dir}}` 下的文件是允许的参考 | `read_memory` 暴露的文件是允许的参考 |
| user `BEGIN` | 每次 lingbot_act 使用完整 task language | 只提供 chunk count |

### 6.3 保留的 prompt

以下基础策略没有被重写：

- `ROLE`
- `CLEAN-TO-RANDOMIZED TRANSFER`
- `ACCURACY-FIRST LOOP`
- `CONDITIONAL TASK-FAMILY PLAYBOOKS`
- `PERCEPTION`
- `BUDGET AND SUCCESS`
- `MODE`

这很重要：compact 不是重新训练一个更笨的策略，而是保持原有的任务分解、视觉验证、
安全边界和失败恢复规则，只改变接口说明。

### 6.4 新 read order

Compact 的 read order 是：

1. `read_memory(ref="guide")`；
2. `view_env_state(step=0)` 并查看 head image；
3. `read_memory(ref="task_info")`；
4. 存在时 `read_memory(ref="task_recipe")`；
5. `read_memory(ref="memory_index")`；
6. 最多读 1–3 个相关 global leaf，例如 `read_memory(ref="grasp_and_hold")`。

它和原版的差别不是阅读内容，而是阅读入口从路径变成了 ref。

### 6.5 新 VLA 规则

Compact CONTROL 的核心句子是：

> `lingbot_act` takes only chunks; the runtime applies the complete current
> task_language and use_length=50.

然后保留 chunk 使用策略：

- 接触、接近成功、状态不稳定、小修正用 1；
- 普通稳定推进用 2；
- 只有 recipe 支持的连续性敏感阶段才用 3；
- VLA 已经有正确接触和可见进展时，不要用推测性 primitive 打断；
- 没有进展时，根据新证据选择继续、缩短下一个 chunk、改善抓取/可见性/物理 staging。

### 6.6 新 RUNTIME 规则

Compact RUNTIME 不再提 `{{memory_dir}}`，而是说：

> Files exposed through `read_memory` are approved planning references and are
> not subject to this restriction.

这保持了安全边界：模型仍然不能读 task source、evaluator、hidden reward 或 expert
trajectory，但可以读白名单 memory。

---

## 7. 一个真实 episode 的前后对照

下面的例子来自 `adjust_bottle_entry2_s100003`，两轮都成功。

### 7.1 原始接口

关键调用序列：

```text
read_text_file("robots/robotwin/guides/GUIDE_RPENT.md")
view_env_state(step=0)
read_text_file("/inspire/.../robotwin_adjust_bottle_demo_randomized_s0.json")
read_text_file("/inspire/.../robotwin_adjust_bottle_demo_randomized_s0_recipe.jsonl")
read_text_file("/inspire/.../memory/MEMORY.md")
query_world_map({"view": "head", "bbox": [140, 80, 200, 240]})
lingbot_act({
  "chunks": 3,
  "prompt": "Pick the medium green bottle upright from the table",
  "use_length": 50
})
finish(...)
```

该 transcript 的统计：

```text
turns_used        = 6
tool_calls        = 8
total_input_tokens  = 118774
total_output_tokens = 848
```

### 7.2 Compact 接口

关键调用序列：

```text
read_memory({"ref": "guide"})
view_env_state(step=0)
read_memory({"ref": "task_info"})
read_memory({"ref": "task_recipe"})
read_memory({"ref": "grasp_and_hold"})
lingbot_act({"chunks": 3})
finish(...)
```

该 transcript 的统计：

```text
turns_used        = 7
tool_calls        = 7
total_input_tokens  = 65954
total_output_tokens = 585
```

这一对 episode 的选择性差异是：

- 输入 token：118774 -> 65954，约 -44.5%；
- 输出 token：848 -> 585，约 -31.0%；
- compact 多一个 turn，但少一次显式工具调用；
- 动作决策相同：都只调用一次 `chunks=3` 的 VLA，然后成功。

这不是说每个 episode 都能省 44%；它说明路径、schema 和重复文本的累积效应可以很
大。主统计使用 47 个双方成功 episode 的配对均值，得到输入 -10.8%、输出 -27.1%。

### 7.3 工具参数长度

在同一批 transcript 中，把 tool_use 的 JSON 参数序列化后统计：

| 工具 | 原始平均参数字符 | Compact 平均参数字符 | 约减少 |
|---|---:|---:|---:|
| `read_text_file` / `read_memory` | 157.2 | 21.3 | 86.5% |
| `lingbot_act` | 131.4 | 12.0 | 90.9% |

这些是“参数字符数”，不是 API 计费 token；但它解释了为什么对话历史会变短。真正的
账单 token 还包含图像、prompt、memory 正文、模型 reasoning 和工具结果。

### 7.4 实际 ref 分布

Compact 批次的 transcript 中，`read_memory` 的 ref 使用大致是：

```text
guide                         70
task_info                     70
task_recipe                   70
grasp_and_hold                39
memory_index                  27
transport_place_release       18
bimanual_and_multi_object     14
perception_and_verification    4
vla_chunk_and_primitive_selection 1
articulated_object             1
```

可以看到，前三个 ref 基本是每个 episode 的固定 read order，后面才是任务相关的
global leaf。compact 并没有减少“要读哪些内容”，只是减少了“怎么指向这些内容”。

---

## 8. 批量评测封装

### 8.1 为什么需要单独的批量脚本

`run_compact_eval.py` 不是重新实现调度器，而是复用 `run_collab_eval.py` 的：

- 任务列表和 seed 生成；
- K8/N4 rule 读取；
- Qwen memory / GPT memory 校验；
- GPT memory 渲染；
- 共享 VLA server 启动；
- 2 GPU × 5 worker 调度；
- timeout、status、resume 和 summary。

它只做三件事：

1. 把 `base.RUNNER` 换成 `run_compact_eval_episode.sh`；
2. 包装 `run_episode()`，在结果中加 `tool_variant=compact_memory_ref_chunks`；
3. 重新写一次 `result.json`，让 compact 标记落盘。

### 8.2 批量配置

本批主实验配置：

```text
worker model       = openai-chat:qwen3.6-27b
worker effort      = none
senior model       = openai:gpt-6-astra
senior effort      = high
task_config        = demo_randomized
entry indexes      = 2,3
planned episodes   = 100
GPU                = 0,1
workers per GPU    = 5
planner timeout    = 1200 s
tool timeout       = 1200 s
subprocess timeout = 1200 s
max turns          = 100
max episode steps  = 10000
```

任务列表来自 `robots/robotwin/eval/demo_randomized.json`。对每个 task 的 entry
`2` 和 `3` 读取对应的 seed 与 task_language；因此一共是 50 × 2 = 100 个
episode。这里的 entry index 不是数组下标以外的随机数，而是 benchmark 文件中预定义
的 seed 条目。

### 8.3 规则和 memory

- Qwen memory 来自 `runs/robotwin-ecc-stage2-gpt-memory/memory`；
- GPT memory 来源是 `memory/robotwin/task-specific/<task>_s0.json`；
- 规则目录是 `runs/robotwin-ecc-stage2-gpt-memory-threshold1-k8-k4/rules`；
- 规则字段为 `gpt_plan`、`checkin_K=8`、`errors_N=4`；
- 每个 episode 的 `result.json` 会记录 rule、模型、memory 路径和 timeout 配置。

`run_collab_eval.py` 会先检查：

- 每个 task 是否有 rule；
- Qwen task-specific `.json` 和 `_recipe.jsonl` 是否存在；
- GPT task-specific audit 是否存在。

任何一项缺失都会在启动前失败，而不是跑到一半才发现。

### 8.4 单 episode 环境变量

`run_compact_eval_episode.sh` 需要以下变量：

```text
TASK_NAME
SEED
OUTPUT_DIR
QWEN_MEMORY_DIR
GPT_MEMORY_FILE
GPU
ROBOTWIN_ASSETS_PATH
INFINI_BASE_URL / INFINI_API_KEY
RPENT_BASE_URL / RPENT_API_KEY
RPENT_ESCALATION_PLAN
RPENT_ESCALATION_CHECKIN
RPENT_ESCALATION_ERRORS
```

如果使用共享 VLA，还会传 `VLA_ENDPOINT`；否则必须传 `LINGBOT_MODEL_PATH`。

脚本内部设置：

```text
--robot robotwin_compact
--planner api
--model openai-chat:qwen3.6-27b
--reasoning-effort none
--memory-profile local
--memory-dir QWEN_MEMORY_DIR
--planner-timeout-s 1200
--max-turns 100
--max-episode-steps 10000
```

并把 GPT senior 配置为：

```text
RPENT_ESCALATION_MODEL=openai:gpt-6-astra
RPENT_ESCALATION_REASONING_EFFORT=high
RPENT_ESCALATION_KEY_ENV=RPENT_API_KEY
RPENT_CLOUD_CORRECTIONS=GPT_MEMORY_FILE
```

### 8.5 Timeout 的三层含义

这套评测里“1200 秒”不是单一的：

1. `--planner-timeout-s` / `CELL_TIMEOUT_S`：agent loop 的 wall-clock 上限；
2. `RPENT_TOOL_TIMEOUT_S`：单次工具执行的超时；
3. `--subprocess-timeout-s`：批量调度器等待整个 episode 子进程的上限。

如果第三层触发，`result.json` 会记录：

```json
{
  "exit_code": 124,
  "subprocess_timeout": true,
  "solved": false
}
```

这会让该 episode 被计为 finished，但通常不会产生完整 transcript，也不一定产生
`finish`。因此 timeout 数和成功率必须一起看，不能只看成功数。

### 8.6 Resume 语义

`run_collab_eval.py` 启动时读取 `run_dir/*/result.json`，已经存在的
`(task, entry_index, seed)` 会被跳过。因此：

- 正常中断后可以重跑同一命令继续剩余 episode；
- 但 timeout 也会写 `result.json`，默认不会自动重试；
- 如果要重试某个 timeout，应删除该 episode 的 `result.json`（以及需要时的旧
  transcript/log），再启动；
- 不要删除整个 `run_dir`，否则会丢失 GPT memory 渲染和已有结果。

### 8.7 主要产物

| 路径 | 内容 |
|---|---|
| `runner.log` | 批量进度，每个 episode 一行 JSON |
| `status.json` | finished/success/timeout/GPT 调用统计 |
| `batch.json` | 本次批量配置和路径 |
| `summary.json` | 最终汇总 |
| `gpt-memory/<task>.md` | 由 GPT audit 渲染的 per-task memory |
| `<task>_entry<n>_s<seed>/result.json` | episode 结构化结果 |
| `escalation.jsonl` | Qwen -> GPT escalation 记录 |
| `transcript_*.json` | planner 消息、usage、finish |
| `run.log` | 工具调用和模型输出的逐步日志 |
| `states.json` | 环境状态、观测和成功标记 |

---

## 9. 复现方法

### 9.1 前置条件

- 已安装本项目依赖，`rpent` 可执行文件可用；
- RoboTwin assets 和 LingBot checkpoint 路径存在；
- Qwen 的 Infini 端点和 key 可用；
- GPT 的 RPENT 端点和 key 可用；
- 至少有 2 张可用于 Env/VLA 的 GPU。

不要把真实 key 写入文档、脚本或 git。下面的命令用环境变量占位：

```bash
export INFINI_BASE_URL=https://cloud.infini-ai.com/maas/
export INFINI_API_KEY=<your-key>
export RPENT_BASE_URL=https://ai.soruxgpt.com
export RPENT_API_KEY=<your-key>
```

### 9.2 单 episode

```bash
TASK_NAME=adjust_bottle \
SEED=100003 \
OUTPUT_DIR=runs/robotwin-compact-smoke \
QWEN_MEMORY_DIR=runs/robotwin-ecc-stage2-gpt-memory/memory \
GPT_MEMORY_FILE=memory/robotwin/adjust_bottle_s0.json \
GPU=0 \
ROBOTWIN_ASSETS_PATH=runs/robotwin-ecc-stage1/assets \
LINGBOT_MODEL_PATH=runs/robotwin-ecc-stage1/models/LingBot-VLA-RoboTwin-EEF-ckpt1500 \
INFINI_BASE_URL="$INFINI_BASE_URL" \
INFINI_API_KEY="$INFINI_API_KEY" \
RPENT_BASE_URL="$RPENT_BASE_URL" \
RPENT_API_KEY="$RPENT_API_KEY" \
RPENT_ESCALATION_PLAN=1 \
RPENT_ESCALATION_CHECKIN=8 \
RPENT_ESCALATION_ERRORS=4 \
bash scripts/ecc_robotwin/run_compact_eval_episode.sh
```

### 9.3 批量评测

```bash
python scripts/ecc_robotwin/run_compact_eval.py \
  runs/robotwin-compact-eval \
  runs/robotwin-ecc-stage2-gpt-memory-threshold1-k8-k4/rules \
  runs/robotwin-ecc-stage2-gpt-memory/memory \
  memory/robotwin \
  --assets-path runs/robotwin-ecc-stage1/assets \
  --model-path runs/robotwin-ecc-stage1/models/LingBot-VLA-RoboTwin-EEF-ckpt1500 \
  --gpus 0,1 \
  --workers-per-gpu 5 \
  --entry-indexes 2,3 \
  --timeout-s 1200 \
  --subprocess-timeout-s 1200
```

### 9.4 续跑

直接重新执行相同命令即可。调度器会跳过已有 `result.json` 的 episode。若需要重试
timeout：

```bash
rm runs/robotwin-compact-eval/<task>_entry<n>_s<seed>/result.json
python scripts/ecc_robotwin/run_compact_eval.py ...
```

### 9.5 结果检查

```bash
cat runs/robotwin-compact-eval/status.json
cat runs/robotwin-compact-eval/summary.json
tail -f runs/robotwin-compact-eval/runner.log
```

检查单个 episode：

```bash
cat runs/robotwin-compact-eval/<episode>/result.json
grep -E '\[tool>\]|\[usage\]|\[model\]' runs/robotwin-compact-eval/<episode>/run.log
```

检查 compact 协议是否真的生效：

```bash
rg 'lingbot_act\(' runs/robotwin-compact-eval/*/run.log
rg 'read_text_file|list_dir' runs/robotwin-compact-eval/*/run.log
rg 'read_memory\(' runs/robotwin-compact-eval/*/run.log
```

预期：

- `lingbot_act` 只出现 `chunks`；
- 不出现 `read_text_file` 或 `list_dir`；
- 出现大量 `read_memory`；
- 不应出现绝对路径形式的 memory ref。

---

## 10. 实测结果

### 10.1 主结果

评测规模：50 个任务 × entry `2,3` = 100 episodes；K8/N4；1200 s timeout。

| 指标 | K8/K4 对照 | Compact | 变化 |
|---|---:|---:|---:|
| 成功数 | 57/100 | **56/100** | -1 |
| 成功率 | 57.0% | **56.0%** | -1.0 pp |
| 成功 episode 平均 GPT 调用 | 0.684 | **0.482** | -29.5% |
| 成功 episode 平均耗时 | 408.3 s | **382.2 s** | -6.4% |
| timeout | 41 | **30** | -11 |
| GPT 调用总数 | 172 | **129** | -25.0% |

Compact 的主结果按 56/100 记录。另有一轮 compact 复跑为 50/100，说明采样和
timeout 会带来波动；在没有更大样本前，不应把单轮差异解释成稳定的算法退化。

### 10.2 配对结果

把同一 `(task, entry, seed)` 的两个配置配对：

| 结果 | Episode 数 |
|---|---:|
| 双方都成功 | 47 |
| 仅 compact 成功 | 9 |
| 仅对照成功 | 10 |
| 双方都失败 | 34 |

双方都成功 47 对，说明主结果不是靠一两个任务偶然拉高。compact 有 9 个独有成功，
对照有 10 个独有成功，差异接近对称。

### 10.3 Token 统计

只统计双方都成功的 47 对，且只统计 Qwen planner usage：

| 指标 | K8/K4 对照 | Compact | 变化 |
|---|---:|---:|---:|
| 输出 token / episode | 1337.5 | **974.7** | -27.1% |
| 输出 token / turn | 176.1 | **125.2** | -28.9% |
| 输入 token / episode | 132920.5 | **118575.9** | -10.8% |
| 输出 token 中位数 / episode | 1141 | **914** | -19.9% |

输出 token 的下降大于输入 token，是因为：

- 每次 `lingbot_act` 不再生成完整 task language；
- 每次 memory 读取不再生成绝对路径；
- 但输入侧仍包含大块 memory 正文、图像和 observation。

### 10.4 工具调用检查

对 compact 主批次做日志/transcript 扫描：

- `lingbot_act` 调用参数 key 只有 `chunks`；
- 没有 `read_text_file` 或 `list_dir`；
- `read_memory` ref 以 `guide`、`task_info`、`task_recipe`、`memory_index` 和
  global stem 为主；
- 没有绝对路径 ref。

这些检查验证的是“协议确实按设计执行”，不是“任务成功”。任务成功仍以
`states.json` 里的 native `eval_success` 为准。

### 10.5 如何解读成功率差异

57 -> 56 的差异只有一个 episode，且配对结果是 47/9/10/34。合理结论是：

- compact 没有造成明显、系统性的精度损失；
- 单轮 100 episode 的随机性和 timeout 足以解释 1 个 episode 的差异；
- 若要声称“严格不降精度”，需要更多 seed 或重复实验；
- 不应把 56 和另一轮 50 的差异解释成 compact 本身的退化。

---

## 11. Token 节约到底来自哪里

### 11.1 输入侧

输入侧减少来自：

1. `read_text_file` 和 `list_dir` 的 schema 被移除；
2. `read_memory` 的 schema 比通用文件工具短；
3. `lingbot_act` 从三个 property 变成一个；
4. system prompt 不再包含 `{{memory_dir}}` 展开后的长路径；
5. 工具调用参数变短，导致后续 request 的历史上下文变短；
6. 工具结果不再回显 path、size、prompt 和 ignore flags。

但输入侧仍有大量不可压缩内容：

- 图像 base64 或 image token；
- 机器人状态；
- memory 正文；
- recipe JSONL；
- world map 查询结果；
- VLA 执行结果。

因此输入 token 只下降约 10.8%，而不是 50%。

### 11.2 输出侧

输出侧减少来自：

- `lingbot_act` 的参数从约 131 字符降到 12 字符；
- memory 参数从约 157 字符降到 21 字符；
- 模型不再重复完整 task language；
- 模型不再生成绝对路径；
- 少一次因旧参数格式导致的重试或解释。

输出 token 的 27% 下降是 compact 最直接、最可解释的收益。

### 11.3 结果侧

`lingbot_act` 每次结果删除三个字段。单次看起来不大，但一个 episode 可能调用 VLA
多次；每次结果都会进入后续上下文。删除的是重复的任务文本和状态 flag，不会删除
动作执行结果。

### 11.4 没有压缩的部分

以下仍然完整保留：

- `view_env_state` 的完整 observation；
- head / wrist 图像；
- robot_state；
- world map 和 camera metadata；
- memory 正文；
- `move_to`、`set_gripper`、`release` 的参数和结果；
- `finish` 的 summary；
- GPT escalation 的 correction 内容。

所以 compact 的边界很清楚：它优化的是“模型如何说”，不是“机器人看到什么”或
“VLA 怎么动”。

---

## 12. 设计选择和替代方案

### 12.1 为什么用符号 ref，而不是更短的 hash

短 hash 更短，但不可读、不可审计，模型也无法从名字推断内容。`task_info`、
`task_recipe`、`grasp_and_hold` 这些名字同时提供语义提示，便于模型选择正确的
memory leaf。当前版本的 ref 长度已经足够短，继续压缩到 hash 的收益有限。

### 12.2 为什么不直接删除 memory 读取

Memory 是评测精度的一部分。删除 memory 会改变任务难度，也会把 compact 变成另一个
baseline。当前方案只改变 memory 的寻址方式，保留正文和 read order。

### 12.3 为什么不用 `read_memory` 代替所有文件工具

`read_memory` 只暴露 curated memory 和 guide，不暴露 task source、evaluator 或
任意 artifact。保留 `read_image` 是因为图像是 observation 的一部分；保留
`write_text_file` 是因为基础 toolkit 的通用行为，不应在 compact 中引入额外语义。

### 12.4 为什么不压缩 `finish`

`finish` 的 summary 是评测审计和失败分析的重要信息。它不参与机器人控制，也不会在
一个 episode 中高频重复。压缩它的收益远小于破坏可读性的风险。

### 12.5 为什么必须同时改 prompt

只改 schema 会让模型持续尝试旧参数，造成无效调用和重试。只改 prompt 而不改 schema
则无法真正阻止模型生成长参数。两者必须同步，才能让“协议约束”和“行为提示”一致。

### 12.6 为什么没有改 exploration

Exploration 需要 `reset`、recipe 写入、memory merge 和多次 attempt；这些行为与
评测阶段的只读 memory、单 episode、no-restart 语义冲突。当前 compact 明确只支持
evaluation，避免把评测优化误用于探索流程。

---

## 13. 限制和风险

### 13.1 协议层限制

- `read_memory` 的 enum 在 toolkit 初始化时生成；新增 global 文件需要新 run。
- `guide` 修补依赖精确字符串匹配，guide 文案改动可能导致替换失效。
- 40,000 字符截断可能影响超长 memory。
- `task_recipe` 中的旧 `prompt`/`use_length` 字段不会被改写。
- `write_text_file` 仍然暴露，虽然 memory 写入受 read-only 约束。

### 13.2 评测层限制

- 1200 s timeout 会截断长尾 episode，成功率和 timeout 必须一起报告。
- timeout 会写 `result.json`，默认续跑不会重试。
- API 延迟、模型采样和 GPU 排队会带来 run-to-run 波动。
- 100 episode 的 1 个差异不足以证明统计显著。
- 成功判定来自 native `eval_success`，不是 `finish` 的 status。

### 13.3 语义层限制

- Compact 依赖“VLA 使用 env task language、use_length 固定 50”这一实现事实。
- 如果未来 VLA 接口改变，compact 的假设需要重新验证。
- 它不减少视觉 token，也不解决长上下文本身。
- 它不是通用工具压缩框架，不能直接套到 LIBERO 或其他 robot。

### 13.4 评测解释限制

- 主结果 56/100 和复跑 50/100 的差异说明单轮噪声不可忽略。
- 成功 episode 的 GPT 调用均值下降，不等于所有 episode 都下降；应同时看失败和
  timeout。
- 输入 token 的配对统计只覆盖双方成功 episode，存在选择偏差；它用于看协议趋势，
  不是完整成本账单。

---

## 14. 如果要把这套方法扩展到别的 robot

可以复用的模式：

1. 找出“运行时已经知道、模型却被迫生成”的字段；
2. 找出“路径/ID 对决策无意义、正文才有意义”的资源；
3. 找出“每次结果都回显、但下一步不需要”的字段；
4. 用白名单 resolver 替代任意路径；
5. 同步修改 prompt，避免旧说明和新 schema 冲突；
6. 保持动作、观测和成功判定不变；
7. 用配对 episode 同时看成功率、timeout 和 token。

不要直接复用的部分：

- RoboTwin 的 `reference_tag` 命名；
- `use_length=50`；
- `chunks` 的 1–3 范围；
- `GUIDE_RPENT.md` 的精确替换；
- Qwen + GPT 的 K8/N4 规则。

---

## 15. 代码阅读顺序

建议按以下顺序阅读：

1. `robots/robotwin_compact/robot_spec.py`
   - 看 `replace()` 如何保留底层 spec；
   - 看 `get_toolkit()` 为什么只允许 evaluation。
2. `robots/robotwin_compact/toolkit.py`
   - 看 `_memory_refs()` 的白名单；
   - 看 `_read_memory()` 的归一化、错误和截断；
   - 看 `_compact_lingbot_act()` 的字段清理。
3. `robots/robotwin_compact/prompts.py`
   - 对比基础 prompt 的四个替换点；
   - 确认没有删除任务策略。
4. `robots/robotwin/primitives.py` 的 `lingbot_act()`
   - 看 task language 实际从哪里读取；
   - 看 chunk 循环、提前结束和 completion。
5. `robots/robotwin/tools.py`
   - 对比原始 `lingbot_act` schema。
6. `scripts/ecc_robotwin/run_compact_eval.py`
   - 看它如何替换 runner 和写 `tool_variant`。
7. `scripts/ecc_robotwin/run_compact_eval_episode.sh`
   - 看环境变量、Qwen/GPT 模型和 timeout。
8. `scripts/ecc_robotwin/run_collab_eval.py`
   - 看任务生成、GPT memory 渲染、并发、resume 和 status。

---

## 16. 最小检查清单

如果要确认一次 compact 评测是否可信，至少检查：

- [ ] `batch.json` 的 entry indexes、timeout、GPU/worker 数与预期一致；
- [ ] `status.json` 的 `planned_run_count` 等于 100；
- [ ] `finished_count` 等于 100；
- [ ] 成功数从 `states.json` 的 native `eval_success` 计算，而不是只看 `finish`；
- [ ] `run.log` 中 `lingbot_act` 只有 `chunks`；
- [ ] 没有 `read_text_file` / `list_dir`；
- [ ] `read_memory` 没有绝对路径 ref；
- [ ] timeout 数与成功率一起报告；
- [ ] 至少给出配对成功/失败矩阵；
- [ ] token 统计说明统计口径和样本范围。

满足这些条件后，才能把 compact 的差异解释为“工具接口精简”的效果，而不是 seed、
timeout 或评测脚本变化造成的假象。
