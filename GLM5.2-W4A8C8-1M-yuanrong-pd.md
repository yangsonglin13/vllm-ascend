# 基于 Yuanrong 的 GLM-5.2 W4A8C8 8机 A2 1M上下文 PD分离部署

## 概述

本指南提供在 8 台 Atlas 800I A2 服务器上部署 **GLM-5.2 W4A8C8** 量化模型、**1M（1024000）上下文**、PD 分离架构（1P1D）并叠加 **Yuanrong Datasystem 作为 KV Pool 后端** 的详细步骤。

PD 分离架构下，Prefill 节点与 Decode 节点各司其职，通过 `MultiConnector` 在同一 `--kv-transfer-config` 中组合多个连接器：

- **KV Pool 外部缓存池**：由 `AscendStoreConnector` + `backend: "yuanrong"` 承担（读 `YR_CONFIG_PATH`），实现跨请求/跨节点的前缀缓存复用，降低重复前缀场景下的首 token 时延。

> **版本说明**：本教程使用 Docker 镜像 `vllm-ascend:v0.23.0rc1`。Yuanrong 客户端配置通过 `YR_CONFIG_PATH` 指向的 `yuanrong.json` 加载，不再使用旧的 `DS_*` 环境变量。

> **当前版本补丁要求**：`v0.23.0rc1` 镜像本身不含 Yuanrong multi-buffer API 改造，需先获取单独提供的 patch 文件打到 `/vllm-workspace/vllm-ascend` 仓库。该 patch 对应 commit `e262020c3`（*perf: use Yuanrong multi-buffer APIs with configurable timeouts*），文件名为 `0001-perf-use-Yuanrong-multi-buffer-APIs-with-configurabl.patch`。建议先将 patch 上传到容器内固定目录，然后执行：
>
> ```bash
> # 配置git用户信息
> git config --global user.email "deploy@local"
> git config --global user.name "deploy"
>
> # vllm-ascend patch（Yuanrong multi-buffer API + JSON 配置改造）
> cd /vllm-workspace/vllm-ascend
> git am /workspace/yuanrong_patches/0001-perf-use-Yuanrong-multi-buffer-APIs-with-configurabl.patch
> ```
>
> 打完后 `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/yuanrong_backend.py` 即使用 `mget_h2d_from_multi_buffers` / `mset_d2h_from_multi_buffers` 并通过 `YR_CONFIG_PATH` 读取 `yuanrong.json`。若环境中已含此改动，可跳过此步骤。

## 环境准备

### 硬件要求

- **8 × Atlas 800I A2 服务器**，每台配备 8 张 NPU 卡（每张 64G 显存）
- 已配置 RoCE 网络以获得最佳性能

### 软件要求

采用模型配套的 Docker 镜像，软件版本与 Docker 镜像内置版本保持一致，确保 HDK、固件等软件在配套范围内。
此外：镜像内置 CANN 9.0.1，HDK 版本要求至少 25.5.0（本指南默认开启 RH2D，通过 RoCE 传输需要 HDK ≥ 25.5.0）。

### 环境信息

| 组件 | 版本 | 备注 |
|------|------|------|
| 服务器硬件 | Atlas 800I A2 × 8 | 4机P + 4机D |
| vLLM-Ascend | vllm-ascend:v0.23.0rc1 | |
| CANN | 9.0.1（镜像内置） | |
| HDK | ≥ 25.5.0 | 本指南默认开启 RH2D，通过 RoCE 需 HDK ≥ 25.5.0 |
| GLM-5.2 权重 | W4A8C8 量化（含 MTP） | [ModelScope](https://www.modelscope.cn/models/ZhipuAI/GLM-5.2) |

### 模型权重

下载 GLM-5.2 W4A8C8 量化模型权重（含 MTP 头），放置到指定目录，如 `/path/to/GLM-5.2-W4A8C8-MTP/`。

> **模型下载地址**：[魔搭社区 GLM-5.2](https://www.modelscope.cn/models/ZhipuAI/GLM-5.2)，W4A8C8 量化版本可使用 [msmodelslim](https://gitcode.com/Ascend/msmodelslim) 自行量化或获取社区已量化版本。

## 使用 Docker 运行

本教程使用的 Docker 镜像版本为 `vllm-ascend:v0.23.0rc1`。如本地尚未下载，可执行以下命令：

```bash
docker pull quay.io/ascend/vllm-ascend:v0.23.0rc1
```

如果下载较慢，可将 `quay.io` 替换为 `m.daocloud.io/quay.io` 或 `quay.nju.edu.cn` 以加速拉取。

### 创建容器

在所有 8 个节点上分别保存同一份 `start-docker.sh`：

```bash
#!/bin/bash
IMAGES_ID="$1"
NAME="$2"

if [ $# -ne 2 ]; then
    echo "error: 需要传入2个参数，格式：$0 <镜像ID> <容器名>"
    exit 1
fi

if ! docker images --format "{{.ID}}" | grep -q "^${IMAGES_ID:0:12}$"; then
    echo "error: 镜像ID $IMAGES_ID 不存在"
    exit 1
fi

docker run --name "${NAME}" -it -d --net=host --shm-size=800g \
    --privileged=true \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /data:/data \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -e http_proxy="$http_proxy" \
    -e https_proxy="$https_proxy" \
    "${IMAGES_ID}"
```

查看镜像 ID：

```bash
docker images | grep vllm-ascend
```

在每个节点分别创建容器：

```bash
# 在每个节点执行（替换为实际镜像ID）
bash start-docker.sh <镜像ID> glm52-yuanrong
```

进入容器：

```bash
docker exec -it glm52-yuanrong bash
```

## 安装 Yuanrong Datasystem

```bash
wget https://gitcode.com/openeuler/yuanrong-datasystem/releases/download/v0.9.2/openyuanrong_datasystem-0.9.2-cp312-cp312-manylinux_2_35_aarch64.whl

pip install openyuanrong_datasystem-0.9.2-cp312-cp312-manylinux_2_35_aarch64.whl
```

验证安装：

```bash
python -c "import yr.datasystem; print('Yuanrong Datasystem 安装成功')"
```

## 安装 etcd

后续 Yuanrong 服务启动脚本依赖 `etcd` 和 `etcdctl`。至少在 P 主节点安装。

```bash
ETCD_VERSION="v3.5.12"
if [ "$(uname -m)" = "aarch64" ]; then
  ETCD_ARCH="linux-arm64"
else
  ETCD_ARCH="linux-amd64"
fi
wget https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
tar -xvf etcd-${ETCD_VERSION}-${ETCD_ARCH}.tar.gz
cd etcd-${ETCD_VERSION}-${ETCD_ARCH}
cp etcd etcdctl /usr/local/bin/
```

验证安装：

```bash
etcd --version
etcdctl version
```

## 启动 Yuanrong 服务

8 机场景需要在所有 8 个节点都启动 Datasystem Worker，并连接同一个 etcd。

整体运行顺序：

```bash
# 节点0 启动ETCD
bash run_etcd.sh

# 节点 0～7 依次启动
bash run_yr_worker.sh
```

### 启动ETCD

创建启动脚本 `run_etcd.sh`，在节点0（P主节点）启动ETCD：

```bash
#!/bin/bash

export ETCD_IP="<P主节点IP>"
export ETCD_PORT=2379
export ETCD_PEER_PORT=2380

etcd \
  --name etcd-single \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:${ETCD_PORT} \
  --advertise-client-urls http://${ETCD_IP}:${ETCD_PORT} \
  --listen-peer-urls http://0.0.0.0:${ETCD_PEER_PORT} \
  --initial-advertise-peer-urls http://${ETCD_IP}:${ETCD_PEER_PORT} \
  --initial-cluster etcd-single=http://${ETCD_IP}:${ETCD_PEER_PORT} \
  > /tmp/etcd.log 2>&1 &

sleep 3

etcdctl --endpoints "${ETCD_IP}:${ETCD_PORT}" put key "value"
etcdctl --endpoints "${ETCD_IP}:${ETCD_PORT}" get key

echo "ETCD start finished, log dir: /tmp/etcd.log"
```

验证：

```bash
# 预期输出：{"health":"true","reason":""}
etcdctl --endpoints "${ETCD_IP}:${ETCD_PORT}" endpoint health
```

### 启动Yuanrong worker

每个节点创建启动脚本 `run_yr_worker.sh`（本指南默认 A2 + 开启 RH2D / RoCE）：

```bash
#!/bin/bash
export HOST_IP="<当前节点IP>"
export ETCD_IP="<ETCD_IP>"
export WORKER_PORT=18481
export ETCD_PORT=2379
export WORKER_LOG_DIR="/var/log/yuanrong/worker"
mkdir -p "${WORKER_LOG_DIR}"

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
    --log_dir "${WORKER_LOG_DIR}" \
    --remote_h2d_device_ids "0,1,2,3,4,5,6,7" \
    > "${WORKER_LOG_DIR}/dscli_start.log" 2>&1 &

echo "Yuanrong service start finished, log dir: ${WORKER_LOG_DIR}"
```

**日志位置**：

- `--log_dir`（对应内部 gflag `log_dir` / `FLAGS_log_dir`）：指定 Datasystem worker 运行日志目录，启动前需 `mkdir -p` 并确保 worker 进程有写权限；可用 `--log_filename` 进一步指定日志文件名（默认文件名通常以 `datasystem_worker` 开头）。
- `dscli start` 的 stdout/stderr 重定向到 `${WORKER_LOG_DIR}/dscli_start.log`，记录启动阶段的输出与错误。
- 客户端侧（vLLM 进程）的 Yuanrong SDK 日志由 `DATASYSTEM_CLIENT_LOG_DIR` 指定（见 [环境变量说明](#环境变量说明)），与 worker 日志分开存放。

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

**停止 Worker**：

```bash
dscli stop --worker_address ${HOST_IP}:${WORKER_PORT}
```

### 配置 yuanrong.json

客户端通过 `YR_CONFIG_PATH` 环境变量指向 `yuanrong.json`，后端启动时读取其作为 Yuanrong 客户端连接配置。在每个 P / D 节点准备同一份 `yuanrong.json`（`worker_addr` 按本节点 IP 修改）：

```json
{
    "worker_addr": "<本节点IP>:18481",
    "connect_timeout_ms": 9000,
    "request_timeout_ms": 0,
    "get_sub_timeout_ms": 0,
    "enable_remote_h2d": true,
    "remote_h2d_transport_backend": "P2P_TRANSFER",
    "enable_fabric_mem": false,
    "enable_dev_mem_pregister": false
}
```

- `worker_addr`：Datasystem worker 地址，**必须**与本节点 `dscli start --worker_address` 一致。
- `enable_remote_h2d`：A2 + RH2D 场景置 `true`。
- `remote_h2d_transport_backend`：A2/RoCE 用 `"P2P_TRANSFER"`，需与 worker 侧 `--remote_h2d_link_type "ROCE"`（默认）对应。
- `enable_fabric_mem` / `enable_dev_mem_pregister`：A2 + `P2P_TRANSFER` 模式下后端始终跳过设备内存预注册，保持默认 `false` 即可。
- 超时三参数（`connect_timeout_ms` / `request_timeout_ms` / `get_sub_timeout_ms`）：`connect_timeout_ms` 要求 ≥ 500；`get_sub_timeout_ms` 可大于 `request_timeout_ms`，Get 路径会自动放大该次 RPC 超时以容纳对象就绪等待。

> 若未设置 `YR_CONFIG_PATH`，vLLM 启动时会直接报错终止，因此**必须配置**。

## PD分离部署（8机、1P1D + Yuanrong + 1M上下文）

### 并行策略

- P节点：DP4，TP8（4机，每机 1 个数据并行副本），DCP8，PCP1
- D节点：DP4，TP8（4机，每机 1 个数据并行副本），DCP8，PCP1

> 1M 上下文通过 `--decode-context-parallel-size 8` + `--cp-kv-cache-interleave-size 128` 启用 DCP 切分；MooncakeConnectorV1 的 `prefill`/`decode` 段需与实际 DP/TP 一致（本配置 P/D 均为 `dp_size: 4, tp_size: 8`）。

### 节点分配

| 节点 | 角色 | IP（示例） | 需要文件 |
|------|------|------------|----------|
| 节点 0 | P 主节点 | 71.10.29.138 | launch_online_dp.py、run_dp_template.sh、server.sh、proxy.sh、load_balance_proxy_server_example.py |
| 节点 1 | P 从节点 | 71.10.29.141 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 2 | P 从节点 | 71.10.29.125 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 3 | P 从节点 | 71.10.29.128 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 4 | D 主节点 | 71.10.29.124 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 5 | D 从节点 | 71.10.29.123 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 6 | D 从节点 | 71.10.29.139 | launch_online_dp.py、run_dp_template.sh、server.sh |
| 节点 7 | D 从节点 | 71.10.29.142 | launch_online_dp.py、run_dp_template.sh、server.sh |

**脚本说明**：
- [`launch_online_dp.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)：每个节点都要有，无需修改
- `run_dp_template.sh`：每个节点根据实际情况修改（P/D 各一份模板见下文）
- [`load_balance_proxy_server_example.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)：仅 P 主节点需要

### P节点

`run_dp_template.sh` 模板，请按实际情况修改 `nic_name`、`local_ip`、权重路径、`MOONCAKE_CONFIG_PATH`、`YR_CONFIG_PATH`：

```bash
#!/bin/bash
rm -rf ~/ascend

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

# 自动获取配置
nic_name=$(ifconfig -a | grep -B1 "$(hostname -I | awk '{print $1}')" | grep -v 'inet' | sed 's/://g' | awk '{print $1}')
local_ip=$(hostname -I | awk '{print $1}')

# 以下环境变量无需修改
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib:/usr/local/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/mooncake:$LD_LIBRARY_PATH
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export PYTHONHASHSEED=0

# Mooncake（P↔D 跨节点 KV 传输，保留）
export MOONCAKE_CONFIG_PATH="/path/to/mooncake.json"

# Yuanrong（KV Pool 外部缓存池）
export YR_CONFIG_PATH="/path/to/yuanrong.json"
export DATASYSTEM_CLIENT_LOG_DIR="/var/log/yuanrong/client"
mkdir -p "${DATASYSTEM_CLIENT_LOG_DIR}"

export HCCL_INTRA_ROCE_ENABLE=1

# vllm起服务配置
vllm serve <MODEL_PATH> \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --prefill-context-parallel-size 1 \
    --decode-context-parallel-size 8 \
    --cp-kv-cache-interleave-size 128 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --seed 1024 \
    --enable-chunked-prefill \
    --served-model-name glm-5 \
    --async-scheduling \
    --max-model-len 1024000 \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.9 \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_producer",
                "kv_port": "30000",
                "kv_connector_extra_config": {
                    "prefill": {
                        "dp_size": 4,
                        "tp_size": 8
                    },
                    "decode": {
                        "dp_size": 4,
                        "tp_size": 8
                    }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "backend": "yuanrong"
                }
            }
        ]
    }
    }' \
    --additional-config '{"enable_flashcomm1": true, "enable_dsa_cp": true, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": false}' \
    --profiler-config \
    '{
        "profiler": "torch",
        "torch_profiler_dir": "/path/to/prof",
        "torch_profiler_with_stack": false
    }' \
    --speculative-config '{"num_speculative_tokens": 1, "method":"deepseek_mtp", "enforce_eager":true}' \
    2>&1 | tee glm.log
```

**server.sh**：P节点 DP4、TP8

```bash
# 71.10.29.138 P 主节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address 71.10.29.138 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.141 P 从节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address 71.10.29.138 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.125 P 从节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address 71.10.29.138 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.128 P 从节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address 71.10.29.138 --dp-rpc-port 10521 --vllm-start-port 6600
```

### D节点

`run_dp_template.sh` 模板，请按实际情况修改 `nic_name`、`local_ip`、权重路径、`MOONCAKE_CONFIG_PATH`、`YR_CONFIG_PATH`：

```bash
#!/bin/bash
rm -rf ~/ascend

export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120

# 自动获取配置
nic_name=$(ifconfig -a | grep -B1 "$(hostname -I | awk '{print $1}')" | grep -v 'inet' | sed 's/://g' | awk '{print $1}')
local_ip=$(hostname -I | awk '{print $1}')

# 以下环境变量无需修改
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_BUFFSIZE=2560
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"

export VLLM_USE_V1=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib:/usr/local/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/mooncake:$LD_LIBRARY_PATH

export PYTHONHASHSEED=0

# Mooncake（P↔D 跨节点 KV 传输，保留）
export MOONCAKE_CONFIG_PATH="/path/to/mooncake.json"

# Yuanrong（KV Pool 外部缓存池）
export YR_CONFIG_PATH="/path/to/yuanrong.json"
export DATASYSTEM_CLIENT_LOG_DIR="/var/log/yuanrong/client"
mkdir -p "${DATASYSTEM_CLIENT_LOG_DIR}"

export HCCL_INTRA_ROCE_ENABLE=1

export ACL_OP_INIT_MODE=1

# vllm起服务配置
vllm serve <MODEL_PATH> \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --prefill-context-parallel-size 1 \
    --decode-context-parallel-size 8 \
    --cp-kv-cache-interleave-size 128 \
    --enable-expert-parallel \
    --enable-prefix-caching \
    --seed 1024 \
    --served-model-name glm-5 \
    --async-scheduling \
    --max-model-len 1024000 \
    --max-num-batched-tokens 128 \
    --trust-remote-code \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.95 \
    --safetensors-load-strategy prefetch \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_consumer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "kv_consumer",
                "kv_port": "30100",
                "kv_connector_extra_config": {
                    "prefill": {
                        "dp_size": 4,
                        "tp_size": 8
                    },
                    "decode": {
                        "dp_size": 4,
                        "tp_size": 8
                    }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port":"0",
                    "load_async": true,
                    "backend": "yuanrong"
                }
            }
        ]
    }
    }' \
    --compilation-config \
    '{
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [4,8,16,24,32,40,48,56,64,96,128,160,192,224,256,298,320,352,384]
    }' \
    --profiler-config \
    '{
        "profiler": "torch",
        "torch_profiler_dir": "/path/to/prof",
        "torch_profiler_with_stack": false
    }' \
    --additional-config '{"enable_flashcomm1": false, "enable_dsa_cp": false, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "fuse_muls_add": true, "multistream_overlap_shared_expert": true, "enable_mc2_hierarchy_comm": false, "enable_sparse_sfa_c8": true, "enable_sparse_li_c8": true, "enable_cpu_binding": true, "recompute_scheduler_enable": true}' \
    --speculative-config '{"num_speculative_tokens": 3, "method":"deepseek_mtp", "enforce_eager":true}' \
    2>&1 | tee glm.log
```

**server.sh**：D节点 DP4、TP8

```bash
# 71.10.29.124 D 主节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 0 --dp-address 71.10.29.124 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.123 D 从节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 1 --dp-address 71.10.29.124 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.139 D 从节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 2 --dp-address 71.10.29.124 --dp-rpc-port 10521 --vllm-start-port 6600

# 71.10.29.142 D 从节点
python launch_online_dp.py --dp-size 4 --tp-size 8 --dp-size-local 1 --dp-rank-start 3 --dp-address 71.10.29.124 --dp-rpc-port 10521 --vllm-start-port 6600
```

> **并行策略说明**：上述 `server.sh` 按本配置（P/D 均 DP4 TP8，每机 1 个 DP 副本）给出参考。1M 上下文 + DCP8 场景下，`--tp-size`、`--dp-size-local`、`--dp-rank-start` 需按实际组网与 DCP 切分调整。`launch_online_dp.py` 内部按 `dp_size_local * tp_size` 分配可见设备。

### proxy.sh

只存在于 P 主节点，在 P/D 节点服务启动成功后执行 `bash proxy.sh > proxy.log &`，根据实际情况修改组网 IP。

```bash
unset http_proxy
unset https_proxy
python load_balance_proxy_server_example.py \
    --port 8000 \
    --host 0.0.0.0 \
    --prefiller-hosts \
        71.10.29.138 \
        71.10.29.141 \
        71.10.29.125 \
        71.10.29.128 \
    --prefiller-ports \
        6600 6600 6600 6600 \
    --decoder-hosts \
        71.10.29.124 \
        71.10.29.123 \
        71.10.29.139 \
        71.10.29.142 \
    --decoder-ports \
        6600 6600 6600 6600
```

### 配置参数说明

#### P节点参数（Yuanrong 相关）

| 参数 | 值 | 说明 |
|------|-----|------|
| kv_connector | MultiConnector | 多连接器组合 |
| kv_role (P节点) | kv_producer | Prefill 节点作为 KV 生产者 |
| AscendStoreConnector backend | yuanrong | KV Pool 使用 Yuanrong 后端 |
| kv_load_failure_policy | recompute | KV 加载失败时重计算 |

#### D节点参数（Yuanrong 相关）

| 参数 | 值 | 说明 |
|------|-----|------|
| kv_role (D节点) | kv_consumer | Decode 节点作为 KV 消费者 |
| AscendStoreConnector backend | yuanrong | KV Pool 使用 Yuanrong 后端 |

### 环境变量说明

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `PYTHONHASHSEED` | 0 | 所有节点必须一致，保证 Yuanrong KV Cache 键一致 |
| `YR_CONFIG_PATH` | yuanrong.json 路径 | KV Pool Yuanrong 后端配置（AscendStoreConnector） |
| `DATASYSTEM_CLIENT_LOG_DIR` | /var/log/yuanrong/client | Yuanrong 客户端 SDK 日志目录 |

### MultiConnector 配置结构说明

`MultiConnector` 的 `--kv-transfer-config` JSON 结构包含两个子连接器，职责不同：

```json
{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer | kv_consumer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeConnectorV1",
                "kv_role": "与顶层一致",
                "kv_port": "30000(P) | 30100(D)",
                "kv_connector_extra_config": {
                    "prefill": { "dp_size": N, "tp_size": M },
                    "decode":  { "dp_size": N, "tp_size": M }
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "与顶层一致",
                "kv_connector_extra_config": {
                    "lookup_rpc_port": "端口号（同机不同 DP 副本需唯一）",
                    "backend": "yuanrong",
                    "load_async": "true（仅 D 节点）"
                }
            }
        ]
    }
}
```

**关键注意事项**：

1. **`MooncakeConnectorV1`（P↔D 传输）保留不变**：负责 Prefill→Decode 的跨节点 KV 传输，读 `MOONCAKE_CONFIG_PATH`。本指南不改动此连接器。
2. **`AscendStoreConnector`（KV Pool）用 `backend: "yuanrong"`**：负责外部前缀缓存池，读 `YR_CONFIG_PATH`。从旧 `backend: "mooncake"` 切换到 `backend: "yuanrong"` 即启用 Yuanrong KV Pool。
3. **`kv_role` 一致性**：顶层和子连接器的 `kv_role` 应保持一致（P 为 `kv_producer`，D 为 `kv_consumer`）。
4. **`kv_port` 区分**：MooncakeConnectorV1 的 `kv_port` 在 P/D 应不同（`30000` vs `30100`）。
5. **`PYTHONHASHSEED`**：所有节点必须设置相同的 `PYTHONHASHSEED=0`。

## 功能验证

服务启动后，验证部署是否成功。

### 测试推理

```bash
curl -H "Accept: application/json" \
    -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "glm-5",
        "messages": [{
            "role": "user",
            "content": "你好，请介绍一下人工智能的未来发展趋势。"
        }],
        "stream": false,
        "ignore_eos": false,
        "temperature": 0,
        "max_tokens": 200
    }' http://localhost:8000/v1/chat/completions
```

## 附录

### Yuanrong性能优化

#### 开启RH2D

本指南默认开启 RH2D（A2 + RoCE / P2P_TRANSFER）。RH2D（Remote Host to Device）是基于昇腾 NPU 的跨节点数据传输机制，支持从远端节点主机侧共享内存到设备侧 HBM 内存的直接传输，可显著提升 KV Cache 跨节点传输性能。

> **前提条件**：HDK 版本需 ≥ 25.5.0，CANN 版本需 ≥ 9.0.1。

**1. 配置大页内存**

见 [启动Yuanrong worker](#启动yuanrong-worker) 中的大页准备步骤。

**2. 服务端（Yuanrong Worker）开启 RH2D**

`run_yr_worker.sh` 已含 `--remote_h2d_device_ids "0,1,2,3,4,5,6,7"` + `--enable_huge_tlb true`，即启用 worker 侧 RH2D。链路默认 `ROCE`（对应客户端 `remote_h2d_transport_backend: "P2P_TRANSFER"`）。

**3. 客户端（vLLM）开启 RH2D**

`yuanrong.json` 中 `enable_remote_h2d: true` + `remote_h2d_transport_backend: "P2P_TRANSFER"` 即启用客户端侧 RH2D。

> 排查 RH2D 环境问题时，可临时把 `enable_remote_h2d` 置 `false`，回退到默认 Datasystem 传输路径。

### 参考资料

- [Yuanrong Datasystem 文档](https://atomgit.com/openeuler/yuanrong-datasystem)
- [etcd 文档](https://etcd.io/docs/)
- [vLLM Ascend 文档](https://docs.vllm.ai/projects/vllm-ascend/)
- [GLM-5.2 部署教程](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/tutorials/models/GLM5.2.html)
- [KV Pool 使用指南](docs/source/user_guide/feature_guide/kv_pool.md)
