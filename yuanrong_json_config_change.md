# Yuanrong 后端配置变更说明：环境变量 → JSON 配置文件

## 背景

在 commit `e262020c3`（*perf: use Yuanrong multi-buffer APIs with configurable timeouts*）中，vLLM-Ascend 的 Yuanrong 后端完成了一次重要改造：

1. **切换到 openyuanrong-datasystem 新引入的同步多缓冲区（multi-buffer）API**：`mget_h2d_from_multi_buffers` / `mset_d2h_from_multi_buffers`，替代原有的 `mget_h2d` / `mset_d2h` + `Blob/DeviceBlobList` 组装逻辑。
2. **客户端配置入口从环境变量改为 JSON 文件**：新增 `YR_CONFIG_PATH` 环境变量指向 `yuanrong.json`，原先的 `DS_WORKER_ADDR`、`DS_ENABLE_EXCLUSIVE_CONNECTION`、`DS_ENABLE_REMOTE_H2D` 等环境变量被移除。
3. **新增可配置超时与传输后端选项**：连接超时、请求超时、Get 子超时、传输后端、FabricMem、设备内存预注册等开关统一收口到 JSON 中。
4. **移除冗余逻辑**：删除了旧的 key 规范化、超过 10000 个 key 的分批传输、A2 设备类型判断等兼容/冗余处理。

本文档聚焦于 **Atlas 800I A2 + 开启 RH2D（RoCE 链路）** 的部署，即 8 机 A2 大 EP PD 分离 + Yuanrong KV Pool 的标准生产配置。A2 通过 RoCE 传输，对应客户端 `remote_h2d_transport_backend: "P2P_TRANSFER"`、worker 侧 `--remote_h2d_link_type "ROCE"`。

---

## 一、配置入口变化总览

| 维度 | 旧方式（环境变量） | 新方式（JSON 文件） |
|------|-------------------|--------------------|
| 配置载体 | 一组 `DS_*` 环境变量 | 一个 `yuanrong.json` 文件 |
| 指定方式 | 直接 `export DS_WORKER_ADDR=...` | `export YR_CONFIG_PATH=/path/to/yuanrong.json` |
| Worker 地址 | `DS_WORKER_ADDR` | JSON 中 `worker_addr` |
| 独占连接 | `DS_ENABLE_EXCLUSIVE_CONNECTION` | **已移除**（不再支持） |
| Remote H2D | `DS_ENABLE_REMOTE_H2D` | JSON 中 `enable_remote_h2d` |
| 超时控制 | 无（使用 SDK 默认值） | JSON 中 `connect_timeout_ms` / `request_timeout_ms` / `get_sub_timeout_ms` |
| 传输后端 | 无 | JSON 中 `remote_h2d_transport_backend` |
| 设备内存预注册 | 由 A2 设备类型自动判断 | JSON 中 `enable_dev_mem_pregister`（默认关闭，opt-in） |

> **保留不变的环境变量**：`PYTHONHASHSEED`（必须所有节点一致）、`DATASYSTEM_CLIENT_LOG_DIR`（客户端日志目录）。这两个与 JSON 无关，继续用环境变量设置。

---

## 二、旧方式：环境变量配置（已废弃）

旧版本在 P / D 节点的 `run_dp_template.sh` 中这样配置 Yuanrong 客户端（开启 RH2D 时）：

```bash
# Yuanrong Datasystem
export DS_WORKER_ADDR="${local_ip}:18481"
#export DS_H2D_MEMCPY_POLICY="direct"
#export DS_D2H_MEMCPY_POLICY="direct"
export DS_ENABLE_REMOTE_H2D=1
```

**问题**：
- 超时参数无法调整，只能用 SDK 默认值；
- Remote H2D 的传输链路类型（HCCS / RoCE）无法在客户端侧区分；
- 设备内存是否预注册由"是否 A2 设备"隐式决定，行为不透明；
- `DS_H2D_MEMCPY_POLICY` / `DS_D2H_MEMCPY_POLICY` 在新后端中已不再读取。

---

## 三、新方式：JSON 配置文件

### 3.1 设置环境变量指向 JSON

在 P / D 节点的 `run_dp_template.sh` 中改为：

```bash
# Yuanrong Datasystem
export YR_CONFIG_PATH="/workspace/yuanrong.json"
export DATASYSTEM_CLIENT_LOG_DIR="/var/log/yuanrong/client"
mkdir -p "${DATASYSTEM_CLIENT_LOG_DIR}"
```

后端启动时通过 `YuanrongConfig.load_from_env()` 读取 `YR_CONFIG_PATH` 指向的 JSON：

```python
config_path = os.getenv("YR_CONFIG_PATH")
if not config_path:
    raise ValueError("The environment variable 'YR_CONFIG_PATH' is not set.")
return YuanrongConfig.from_file(config_path)
```

> 若未设置 `YR_CONFIG_PATH`，vLLM 启动时会直接报错终止，因此**必须配置**。

### 3.2 yuanrong.json 模板（A2 + RH2D / RoCE）

```json
{
    "worker_addr": "1.2.3.4:18481",
    "connect_timeout_ms": 9000,
    "request_timeout_ms": 0,
    "get_sub_timeout_ms": 0,
    "enable_remote_h2d": true,
    "remote_h2d_transport_backend": "P2P_TRANSFER",
    "enable_fabric_mem": false,
    "enable_dev_mem_pregister": false
}
```

- `worker_addr`：Datasystem worker 地址，`<host>:<port>` 格式，**必须**与本节点 `dscli start --worker_address` 一致。
- `connect_timeout_ms`：客户端建立连接的超时（毫秒），Yuanrong 要求 ≥ 500，默认 `9000`。
- `request_timeout_ms`：客户端请求超时（毫秒），`0` 表示沿用 SDK 行为（用 `connect_timeout_ms` 作为请求超时），设为正数则独立控制。
- `get_sub_timeout_ms`：每次 `mget_h2d_from_multi_buffers` 等待对象就绪的最大时间（毫秒），`0` 表示不等；**可大于** `request_timeout_ms`，Get 路径会自动放大该次 RPC 超时。
- `enable_remote_h2d`：A2 + RH2D 场景置 `true`。
- `remote_h2d_transport_backend`：A2/RoCE 用 `"P2P_TRANSFER"`，需与 worker 侧 `--remote_h2d_link_type "ROCE"` 对应。
- `enable_fabric_mem` / `enable_dev_mem_pregister`：A2 + `P2P_TRANSFER` 模式下后端始终跳过设备内存预注册，这两个保持默认 `false` 即可。

---

## 四、worker 侧（dscli）启动示例

worker 侧通过 `--remote_h2d_device_ids` 启用 RH2D，链路默认即 `ROCE`（可省略 `--remote_h2d_link_type`）。对应客户端 `yuanrong.json` 设 `enable_remote_h2d: true` + `remote_h2d_transport_backend: "P2P_TRANSFER"`。

```bash
#!/bin/bash
export HOST_IP="<当前节点IP>"
export ETCD_IP="<ETCD_IP>"
export WORKER_PORT=18481
export ETCD_PORT=2379

dscli start \
    --interleave 0-7 \
    --timeout 600 \
    -w \
    --worker_address ${HOST_IP}:${WORKER_PORT} \
    --etcd_address ${HOST_IP}:${ETCD_PORT} \
    --shared_memory_size_mb 409600 \
    --node_timeout_s 300 \
    --node_dead_timeout_s 600 \
    --enable_huge_tlb true \
    --arena_per_tenant 1 \
    --enable_fallocate false \
    --shared_memory_populate true \
    --rpc_thread_num 64 \
    --sc_regular_socket_num 0 \
    --sc_stream_socket_num 0 \
    --oc_thread_num 64 \
    --enable_worker_worker_batch_get true \
    --liveness_check_path /workspace/liveness \
    --remote_h2d_device_ids "0,1,2,3,4,5,6,7"
```

> **大页准备**：开启 `--enable_huge_tlb true` 前必须先分配足够 2 MiB 大页。以 400 GB 共享内存为例，至少需要 200000 页（400 GB ÷ 2 MB）：
>
> ```bash
> # 分配（root）
> echo 200000 > /proc/sys/vm/nr_hugepages
> # 验证
> grep -E "HugePages_Total|HugePages_Free|Hugepagesize" /proc/meminfo
> ```
>
> `--timeout`、`--interleave` 等需放在 `-w` 之前，因为 `-w` 会消费后续所有参数。

---

## 参考资料

- commit `e262020c3` — *perf: use Yuanrong multi-buffer APIs with configurable timeouts*
- commit `26fb5655c` — *perf: add load_kvc and store_kvc latency stats*
- [KV Pool 使用指南](docs/source/user_guide/feature_guide/kv_pool.md)
- [Yuanrong Datasystem 文档](https://atomgit.com/openeuler/yuanrong-datasystem)
