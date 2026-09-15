# benchmark_config.json 配置参数说明

本文档对 `Config/benchmark_config.json` 中的所有配置段和参数进行解析说明。

Benchmark 模式用于运行标准 MovingAI MAPF 基准测试（纯 start→goal 多智能体路径规划，无货架/订单/任务链），通过 `python main.py --benchmark` 启用。

---

## 1. `benchmark` — MAPF 基准测试配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | `bool` | `false` | 是否启用 benchmark 模式（需同时传入 `--benchmark` 命令行参数） |
| `map_path` | `str` | `""` | MovingAI `.map` 文件路径（必填），如 `"Env/maps/warehouse-10-20-10-2-1.map"` |
| `scen_path` | `str` | `""` | MovingAI `.scen` 文件路径。留空时随机生成 agent 的 start/goal 位置 |
| `num_agents` | `int` | `10` | agent 数量。使用 `.scen` 文件时取前 N 个 scenario；随机模式时生成 N 个 agent |
| `max_ticks` | `int` | `500` | 仿真超时 tick 数。超时后未到达目标的 agent 视为失败 |
| `random_seed` | `int` | `42` | 随机生成 start/goal 位置时的随机种子，保证实验可复现 |

---

## 2. `policies` — 策略选择

Benchmark 模式仅使用 `path_planner` 字段，其余策略字段不生效。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path_planner` | `str` 或 `obj` | `"AStarPathPlanner"` | 路径规划算法。支持两种写法（见下方） |

### 写法一：仅指定算法名

```json
"policies": {
    "path_planner": "AStarPathPlanner"
}
```

### 写法二：指定算法名 + 参数

```json
"policies": {
    "path_planner": {
        "name": "PrioritizedPathPlanner",
        "params": {
            "max_horizon": 100,
            "goal_reserve": 6
        }
    }
}
```

### 当前可用 path_planner

| 算法名 | 说明 | 可配参数 |
|--------|------|---------|
| `AStarPathPlanner` | 单智能体 A* 算法 | — |
| `PrioritizedPathPlanner` | 优先级规划（时空 A* + 预留表） | `max_horizon`（搜索深度），`goal_reserve`（目标占用缓冲） |

> 后续接入的 EECBS / LNS2 / LaCAM2 / PIBT 等 planner 注册后即可在此切换。

---

## 3. Agent 生成策略

### 从 .scen 文件加载

当 `scen_path` 非空时，从 MovingAI `.scen` 文件读取 agent 起点和终点。

`.scen` 文件为 tab 分隔格式，每行一个 agent：

```
version 1
bucket  map_name  width  height  start_col  start_row  goal_col  goal_row  optimal_length
0       map.map   32     32      5          10         20        25        18.72
```

系统取前 `num_agents` 行作为实验 agent。

### 随机生成

当 `scen_path` 为空时，系统在地图的 free cell 上随机采样：
- 使用 `random_seed` 初始化随机数生成器
- 采样 `num_agents × 2` 个不重复的 free cell
- 前半作为 start 位置，后半作为 goal 位置

> 同一 `random_seed` + 同一 `map_path` + 同一 `num_agents` 保证生成完全相同的 agent 配置。

---

## 4. 输出指标

运行结束后自动打印以下指标：

| 指标 | 说明 |
|------|------|
| `Agents` | 参与实验的 agent 总数 |
| `Completed` | 在 `max_ticks` 内到达目标的 agent 数 |
| `Success rate` | `Completed / Agents`，以百分比显示 |
| `Makespan` | 最后一个 agent 到达目标的 tick 数（即总完成时间） |
| `Total conflicts` | 仿真过程中检测到的冲突总数（顶点冲突 + 交换冲突） |

---

## 5. 完整配置示例

```json
{
    "benchmark": {
        "enabled": true,
        "map_path": "Env/maps/warehouse-10-20-10-2-1.map",
        "scen_path": "",
        "num_agents": 20,
        "max_ticks": 1000,
        "random_seed": 42
    },
    "policies": {
        "path_planner": {
            "name": "AStarPathPlanner",
            "params": {}
        }
    }
}
```

运行命令：

```bash
python main.py --benchmark --config Config/benchmark_config.json
```

输出示例：

```
[INFO] Loaded map: Env/maps/warehouse-10-20-10-2-1.map (63x161)
[INFO] Generated 20 random agents (seed=42)
[INFO] Path planner: AStarPathPlanner
============================================================
MAPF BENCHMARK RESULTS
  Map:              Env/maps/warehouse-10-20-10-2-1.map
  Planner:          AStarPathPlanner
  Agents:           20
  Completed:        20/20
  Success rate:     100.00%
  Makespan:         188 ticks
  Total conflicts:  3
============================================================
```

---

## 6. 可用地图一览

`Env/maps/` 目录下包含 33 个标准 MovingAI 地图：

| 类别 | 地图文件 | 尺寸 |
|------|---------|------|
| **空地** | `empty-8-8.map` | 8×8 |
| | `empty-16-16.map` | 16×16 |
| | `empty-32-32.map` | 32×32 |
| | `empty-48-48.map` | 48×48 |
| **随机障碍** | `random-32-32-10.map` | 32×32（10% 障碍） |
| | `random-32-32-20.map` | 32×32（20% 障碍） |
| | `random-64-64-10.map` | 64×64（10% 障碍） |
| | `random-64-64-20.map` | 64×64（20% 障碍） |
| **迷宫** | `maze-32-32-2.map` | 32×32 |
| | `maze-32-32-4.map` | 32×32 |
| | `maze-128-128-1.map` | 128×128 |
| | `maze-128-128-2.map` | 128×128 |
| | `maze-128-128-10.map` | 128×128 |
| **房间** | `room-32-32-4.map` | 32×32 |
| | `room-64-64-8.map` | 64×64 |
| | `room-64-64-16.map` | 64×64 |
| **仓库** | `warehouse-10-20-10-2-1.map` | 63×161 |
| | `warehouse-10-20-10-2-2.map` | 63×161 |
| | `warehouse-20-40-10-2-1.map` | 123×321 |
| | `warehouse-20-40-10-2-2.map` | 123×321 |
| **城市** | `Berlin_1_256.map` | 256×256 |
| | `Boston_0_256.map` | 256×256 |
| | `Paris_1_256.map` | 256×256 |
| **游戏** | `brc202d.map`, `den312d.map`, `den520d.map`, `lak303d.map`, `orz900d.map`, `ost003d.map` 等 | 各异 |
