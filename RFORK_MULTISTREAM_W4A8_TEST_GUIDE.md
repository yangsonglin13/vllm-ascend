# RFork + multistream_overlap_shared_expert + W4A8 验证指南

本指南用于验证两个 RFork bug 修复（分支 `fix/rfork-multistream-w4a8-fallback`）：

- **Bug #1**（commit `7c5574cc7`）：RFork 预处理阶段对空权重跑 shared-expert 一致性校验，产生 NaN 并中止传输。
- **Bug #2**（commit `7f5968921`）：RFork fallback 在同进程内重建模型，撞 `DeepseekV32IndexerCache` 等的 `Duplicate layer name` 注册表，导致 worker 挂掉。

适用模型：GLM-5 W4A8（`GlmMoeDsaForCausalLM`，`DeepseekV2ForCausalLM` 子类）+ `--quantization ascend` + `--load-format rfork` + `multistream_overlap_shared_expert: true` + MTP。

---

## 0. 前置准备

### 0.1 切到修复分支并安装

```bash
cd /vllm-workspace/vllm-ascend
git fetch origin
git checkout fix/rfork-multistream-w4a8-fallback
git log --oneline -3
# 应看到:
#   7f5968921 fix(rfork): clear process-global layer registries before fallback re-init
#   7c5574cc7 fix(rfork): skip shared-expert consistency check during pre-transfer post-process
#   b3b2f84ac A3 Decode
```

### 0.2 跑单测和 lint

```bash
pip install -e .[dev]
pytest -sv tests/ut/model_loader/rfork/test_rfork_loader.py
bash format.sh ci
```

单测全绿 + lint 通过是继续实测的前提。

### 0.3 准备 planner

```bash
python examples/rfork/rfork_planner.py --host 0.0.0.0 --port 8000
```

> 每次改配置/换 commit 后，**重启 planner** 或换一个 `model_deploy_strategy_name`，避免 receiver 匹配到 stale seed。

### 0.4 关键概念

| 角色 | 说明 |
|---|---|
| **seed 实例** | 第一个正常加载完成的 vLLM 实例，加载完后向 planner 注册自己，供后续 receiver 复用 |
| **receiver 实例** | 后续启动的实例，从 planner 申请一个 seed，走 RFork 从 seed 拉权重 |

- **Bug #1 只在 receiver 端触发**（seed 走默认 loader，权重真实，校验正常）。
- **Bug #2 在 receiver 端的 fallback 路径触发**（需要先经过 `initialize_model` 填充注册表，再 fallback 重建）。
- **没有可用 seed 时**，receiver 在 `is_seed_available()` 就返回 False，**不会**经过 `initialize_model`，**测不到 Bug #2**。

---

## 1. 基线测试：纯存储加载（脱开 RFork + 多流）

**目的**：确认 GLM-5 W4A8 + MTP 本身能正常加载，排除模型/权重/环境问题。

### 步骤

临时修改 `start_decode_a3.sh`：

1. 删掉 `--load-format rfork`
2. 删掉 `--model-loader-extra-config "${RFORK_CONFIG}"`
3. `additional-config` 里把 `multistream_overlap_shared_expert` 改成 `false`

```bash
bash start_decode_a3.sh
```

### 预期

- 正常加载，能出 token。
- 日志出现 `FusedMoE shared experts split computation matches the integrated computation.`

### 判定

| 结果 | 含义 |
|---|---|
| 正常 serve | 基线 OK，继续往下 |
| 这步就挂 | 不是 RFork/多流的问题，先解决模型/权重/环境，再继续 |

---

## 2. 验证 Bug #1：RFork + 多流 + W4A8 不再爆 NaN

**目的**：确认 commit `7c5574cc7` 修复了"预处理阶段校验跑在空权重上"的问题。

### 2.1 切到只含 Bug #1 的 commit

```bash
git checkout 7c5574cc7
```

### 2.2 恢复 start_decode_a3.sh 到原始配置

- 保留 `--load-format rfork`
- 保留 `--model-loader-extra-config "${RFORK_CONFIG}"`
- `additional-config` 里 `multistream_overlap_shared_expert: true`
- 保留 `--quantization ascend` 和 MTP `--speculative-config`

### 2.3 起 seed 实例（机器 A）

```bash
# 机器 A,用原始 start_decode_a3.sh
bash start_decode_a3.sh
# 等待,直到日志出现:
#   "Seed service started for device_id=..., port=..."
# 此时 A 已向 planner 注册自己作为 seed
```

### 2.4 起 receiver 实例（机器 B，相同配置）

```bash
# 机器 B,同样的 start_decode_a3.sh(相同 model_url / deploy_strategy_name / tp / ep)
bash start_decode_a3.sh
```

### 2.5 看 receiver 日志

**Bug #1 修复的判定标准**：

| 日志特征 | 含义 |
|---|---|
| `RFork uses post-load tensor layout transfer for quantized model.` | 进入预处理路径（Bug #1 修复点） |
| **不再出现** `FusedMoE shared experts split computation does not match the integrated computation.` | Bug #1 修好了 ✅ |
| **不再出现** `Max absolute difference: nan` / `Integrated output - sum: nan` | Bug #1 修好了 ✅ |
| `transfer weights starts, weights: N, chunks: M, total bytes: X GiB` | RFork 传输开始 |
| `transfer weights time: ...` | 传输成功 → 实例正常 serve ✅ |

### 2.6 判定

| 结果 | 含义 |
|---|---|
| 传输成功，receiver 正常 serve | Bug #1 验证通过 ✅ |
| 仍报 NaN + `FusedMoE shared experts split computation does not match` | Bug #1 未修好，需排查 |
| 传输失败进 fallback，撞 `Duplicate layer name` | Bug #1 修好了（没报 NaN），但 Bug #2 存在（预期，因为此 commit 不含 Bug #2 修复） |

---

## 3. 验证 Bug #2：fallback 不再撞 Duplicate layer name

**目的**：确认 commit `7f5968921` 修复了"同进程重建模型撞进程级注册表"的问题。

需要**故意触发 fallback**。最可控的方法：临时让 `RForkWorker.transfer` 直接返回 False。

### 3.1 前提

- planner 中**必须有一个真实可用的 seed**（即第 2.3 步的 seed 实例还在运行）。
- 否则 receiver 在 `is_seed_available()` 就返回 False，**不会**经过 `initialize_model`，`static_forward_context` 是空的，fallback 不会撞 Duplicate layer name——**测不到 Bug #2**。

### 3.2 复现 Bug #2（只含 Bug #1 的 commit）

```bash
# 机器 B
git checkout 7c5574cc7

# 临时让 transfer 失败:在 transfer 方法开头加 return False
# 编辑 vllm_ascend/model_loader/rfork/rfork_worker.py
# 找到 def transfer(self, model) -> bool:
# 在方法体第一行加:    return False
```

修改后的 `transfer` 方法应类似：

```python
def transfer(self, model) -> bool:
    return False   # ← 临时加,验证完删掉
    try:
        assert self.transfer_backend.is_initialized()
        ...
```

```bash
bash start_decode_a3.sh   # 机器 B
```

**看 receiver 日志，应复现 Bug #2**：

| 日志特征 | 含义 |
|---|---|
| `RFork uses post-load tensor layout transfer for quantized model.` | 预处理跑了（注册表已填充） |
| `RFork transfer failed: transfer failed., clean up and fall back to default loader` | 进 fallback |
| `ValueError: Duplicate layer name: model.layers.0.self_attn.indexer.k_cache` | **Bug #2 复现** ❌ |
| `WorkerProc failed to start.` | worker 挂了 ❌ |

### 3.3 验证 Bug #2 修复（含两个 fix 的 commit）

```bash
# 机器 B
git checkout 7f5968921
# 保留上一步加的 return False(不要还原!)
bash start_decode_a3.sh   # 机器 B
```

**看 receiver 日志，应验证 Bug #2 修复**：

| 日志特征 | 含义 |
|---|---|
| `RFork transfer failed: ..., clean up and fall back to default loader` | 进 fallback |
| **不再出现** `Duplicate layer name: model.layers.0.self_attn.indexer.k_cache` | Bug #2 修好了 ✅ |
| **不再出现** `WorkerProc failed to start.` | worker 不挂 ✅ |
| 随后 `Loading model weights took ... seconds` | fallback 从存储加载成功 ✅ |
| 实例最终能 serve | ✅ |

### 3.4 还原临时改动

```bash
# 机器 B,验证完后还原
git checkout vllm_ascend/model_loader/rfork/rfork_worker.py
```

### 3.5 判定

| 结果 | 含义 |
|---|---|
| `7c5574cc7` 撞 Duplicate layer name + worker 挂；`7f5968921` 不挂、fallback 成功 | Bug #2 验证通过 ✅ |
| 两个 commit 都不撞 Duplicate layer name | 可能没经过 `initialize_model`（seed 不可用），检查 3.1 前提 |

---

## 4. 回归测试：正常 RFork 流程不受影响

**目的**：确认修复没有破坏正常 RFork 传输路径。

### 4.1 切到含两个 fix 的 commit 并还原临时改动

```bash
git checkout 7f5968921
git status   # 确认 rfork_worker.py 没有未提交改动(return False 已还原)
```

### 4.2 正常起 seed + receiver

- 机器 A：seed 实例（正常配置）
- 机器 B：receiver 实例（相同配置，`deploy_strategy_name` 一致）

### 4.3 看日志

**预期**：

- receiver 不报 NaN（Bug #1 屏蔽了预处理校验）
- `transfer weights starts` → `transfer weights time: ...`（传输成功，不进 fallback）
- seed 端日志出现 `FusedMoE shared experts split computation matches the integrated computation.`（seed 走默认 loader，校验正常跑）
- 实例能正常 serve

### 4.4 推理正确性校验

跑几个 prompt，对比 seed 端和 receiver 端输出应一致：

```bash
# 在 receiver 端
curl http://<receiver_ip>:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM-5.1-w4a8-modelscope","prompt":"你好","max_tokens":50,"temperature":0}'
```

同样请求打 seed 端，输出应一致（温度 0 保证确定性）。

### 4.5 判定

| 结果 | 含义 |
|---|---|
| 传输成功 + 推理正确 | 回归通过 ✅ |
| 传输失败 | 检查 planner 是否有 stale seed，或 seed/receiver 配置是否一致 |

---

## 5. 验证 checklist 汇总

| 场景 | commit | `return False` | 预期结果 |
|---|---|---|---|
| 基线（无 RFork/无多流） | 任意 | 无 | 正常 serve |
| Bug #1：正常 RFork + 多流 + W4A8 | `7c5574cc7` | 无 | 不报 NaN，传输成功 |
| Bug #2 复现：故意触发 fallback | `7c5574cc7` | 有 | 撞 Duplicate layer name，worker 挂 |
| Bug #2 修复：故意触发 fallback | `7f5968921` | 有 | 不挂，fallback 从存储加载成功 |
| 回归：正常 RFork | `7f5968921` | 无 | 传输成功，推理正确 |

---

## 6. 注意事项

### 6.1 planner 清理

每次改配置/换 commit 后，**重启 planner** 或换一个 `model_deploy_strategy_name`，否则 receiver 匹配到 stale seed 会传输对不上（layout 不一致），可能被误判成 Bug #1 没修好——其实是 seed 端布局变了。

### 6.2 Bug #1 的 commit 单测对照

`7c5574cc7` 在正常传输成功时**不会进 fallback**，所以不会撞 Bug #2。只有**故意让 transfer 失败**才能在 `7c5574cc7` 上看到 Duplicate layer name。别用"正常传输成功"来下"Bug #2 不存在"的结论。

### 6.3 seed 端不会报 Bug #1 的 NaN

seed 走默认 loader，`load_weights` 填真权重后才跑校验。只有 receiver（走 RFork 预处理）才会触发 Bug #1。看日志要盯 receiver 那几个 rank。

### 6.4 没有 seed 时测不到 Bug #2

`is_seed_available()` 返回 False 时，receiver 在 `load_model` 里更早抛 `seed is not available`，**此时 `initialize_model` 还没跑**，`static_forward_context` 是空的，fallback 不会撞 Duplicate layer name。验证 Bug #2 前必须确保有一个真实可用的 seed。

### 6.5 W4A8 + MTP + RFork 是未验证组合

此组合不在 [rfork.md Tested Models](docs/source/user_guide/feature_guide/rfork.md) 表里。首次跑通后建议做一次基本推理校验（同 prompt 在 seed 和 receiver 上输出一致），再信任修复。

### 6.6 单机资源

单机起两个 GLM-5 W4A8（TP8）实例内存可能紧张。两台机器更稳。单机做法：实例 A `--port 8080`，实例 B `--port 8081`，指向同一个 planner，相同拓扑。

---

## 7. 故障排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| receiver 报 `seed is not available` | planner 没有可用 seed | 先起 seed 实例并等它注册 |
| receiver 报 `Cannot get transfer engine session or weight info` | seed service 连不上 / seed_ip 错 | 检查 seed 实例是否存活、网络是否通 |
| 传输成功但推理输出乱码 | seed/receiver layout 不一致 | 换 `model_deploy_strategy_name`，重启 planner 和两端 |
| `7f5968921` 上 fallback 仍撞 Duplicate layer name | 修复未生效 | 确认 `git log` 看到 `7f5968921` 在 HEAD；确认未用旧代码 |
| 单测失败 | 环境问题 | 确认 `pip install -e .[dev]`，检查 vllm 版本 |

---

## 8. 修复原理速览

### Bug #1（commit `7c5574cc7`）

**根因**：`AscendFusedMoE` 包装了 `process_weights_after_loading`，在权重处理完后跑一次 shared-expert 一致性校验。RFork 的 processed-layout-transfer 路径在**从 seed 拉权重之前**先跑 `process_weights_after_loading` 建布局，此时权重是空的 → 校验 forward 产 NaN → 抛 `ValueError`。

**修复**：用 contextmanager [`_shared_expert_consistency_check_disabled`](vllm_ascend/model_loader/rfork/rfork_loader.py) 在预处理那次 `process_weights_after_loading` 期间，把每个 `AscendFusedMoE` 实例的 `_validate_shared_expert_consistency` 临时替换成 no-op，退出自动恢复。普通加载路径不受影响。

### Bug #2（commit `7f5968921`）

**根因**：RFork fallback 在同进程内 `del model` + 重新 `get_model`。`initialize_model` 会在每个 layer `__init__` 里把 prefix 注册进 `compilation_config.static_forward_context`（进程级 dict，不在 module 实例上），`del model` 清不掉它。重建时同 prefix 再注册 → `DeepseekV32IndexerCache.__init__` 抛 `Duplicate layer name` → worker 挂。GLM-5 是 `DeepseekV2ForCausalLM` 子类，复用这套 Indexer/MLA 栈，所以同样中招。

**修复**：新增 [`_reset_process_global_model_state`](vllm_ascend/model_loader/rfork/rfork_loader.py)，仿上游 `vllm.v1.worker.gpu.shutdown.free_before_shutdown`，在 fallback 重建前清掉 `static_forward_context`、`static_all_moe_layers`、`_ROPE_DICT`。各清理带 `getattr`/`try-except` 守护，跨 vLLM 版本兼容。

---

## 9. 提交信息

| Commit | 标题 | 文件改动 |
|---|---|---|
| `7c5574cc7` | `fix(rfork): skip shared-expert consistency check during pre-transfer post-process` | `rfork_loader.py` +80/-1，`test_rfork_loader.py` +43 |
| `7f5968921` | `fix(rfork): clear process-global layer registries before fallback re-init` | `rfork_loader.py` +38，`test_rfork_loader.py` +66 |

两个 commit 均 `-s` sign-off，符合 AGENTS.md。分支 `fix/rfork-multistream-w4a8-fallback`，基于 `v0.20.2rc1-rfork`。
