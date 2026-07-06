#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  start_decode_a3.sh <node-rank> [-- VLLM_ARGS...]

The default topology is one A3 node with 16 NPU DIEs:
  global DP size: 16
  local DP size:  16
  TP size:         1

Examples:
  # One A3 decode instance (DP ranks 0-15)
  POD_IP=10.0.0.20 ./start_decode_a3.sh 0

  # RFork with async model mount fallback
  ENABLE_RFORK=1 \
  ENABLE_ASYNC_MODEL_MOUNT=1 \
  PLANNER_URL=http://10.0.0.10:1223 \
  MODEL_PATH=/home/admin/model-bootstrap \
  RFORK_MODEL_URL=/home/admin/model \
  MOCK_MODEL_WEIGHT_PATH=/home/admin/model-mounted \
  ./start_decode_a3.sh 0

Important environment variables:
  MODEL_PATH       Model directory (default: /home/admin/model/)
  DP_MASTER_ADDR   Decode DP master IP (default: local IP)
  DP_SIZE          Global DP size (default: 16)
  DP_LOCAL_SIZE    Local DP workers per A3 node (default: 16)
  DP_RPC_PORT      Distributed DP RPC port (default: 14435)
  KV_PORT          Mooncake KV port (default: 30400)
  ENGINE_ID        Same Mooncake engine ID on all decode nodes (default: 4)
  ENABLE_RFORK     Set to 1 to use RFork model loading (default: 0)
  PLANNER_URL      RFork planner URL, for example http://10.0.0.10:1223
  MODEL_DEPLOY_STRATEGY_NAME
                    Must match between the seed and receiver decode clusters
  ENABLE_ASYNC_MODEL_MOUNT
                    Set to 1 to race RFork seed retries against async mount
  BOOTSTRAP_MODEL   Metadata/tokenizer model directory used during startup
  FULL_MODEL        Full model directory returned after mount; also the
                    default RFORK_MODEL_URL in async-mount mode
  READY_FILE        Optional alias for MOCK_MODEL_READY_FILE
  MODEL_MANAGER_ROOT
                    Parent directory of model_manager/ (default:
                    /a3_inference/itask/workdir/shared/ysl/async_model_mount_mock)
  MOCK_MODEL_WEIGHT_PATH
                    Full model directory returned after async mount completes
  MOCK_MODEL_MOUNT_DELAY_SEC
                    Optional delay before the mock returns the full model path
  MOCK_MODEL_READY_FILE
                    Optional file gate used to control when async mount wins
  DRY_RUN=1        Print the command without executing it
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

detect_local_ip() {
    local detected_ip
    detected_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [[ -n "${detected_ip}" ]] || die "cannot detect LOCAL_IP; set LOCAL_IP or POD_IP explicitly"
    printf '%s' "${detected_ip}"
}

run_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        exec "$@"
    fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
[[ $# -ge 1 ]] || {
    usage >&2
    exit 2
}

NODE_RANK="$1"
shift
if [[ "${1:-}" == "--" ]]; then
    shift
fi
[[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || die "node rank must be a non-negative integer"

MODEL_PATH="${MODEL_PATH:-${BOOTSTRAP_MODEL:-/home/admin/model/}}"
PORT="${PORT:-8100}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-9100}"
DP_SIZE="${DP_SIZE:-16}"
DP_LOCAL_SIZE="${DP_LOCAL_SIZE:-16}"
DP_RPC_PORT="${DP_RPC_PORT:-14435}"
KV_PORT="${KV_PORT:-30400}"
ENGINE_ID="${ENGINE_ID:-4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-auto}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-${NET_CARD_NAME:-eth0}}"
VLLM_BIN="${VLLM_BIN:-vllm}"
ENABLE_RFORK="${ENABLE_RFORK:-0}"
ENABLE_ASYNC_MODEL_MOUNT="${ENABLE_ASYNC_MODEL_MOUNT:-0}"

[[ "${ENABLE_RFORK}" == "0" || "${ENABLE_RFORK}" == "1" ]] || die "ENABLE_RFORK must be 0 or 1"
[[ "${ENABLE_ASYNC_MODEL_MOUNT}" == "0" || "${ENABLE_ASYNC_MODEL_MOUNT}" == "1" ]] ||
    die "ENABLE_ASYNC_MODEL_MOUNT must be 0 or 1"
if [[ "${ENABLE_ASYNC_MODEL_MOUNT}" == "1" && "${ENABLE_RFORK}" != "1" ]]; then
    die "ENABLE_ASYNC_MODEL_MOUNT=1 requires ENABLE_RFORK=1 for this RFork test"
fi

[[ "${DP_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "DP_SIZE must be a positive integer"
[[ "${DP_LOCAL_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "DP_LOCAL_SIZE must be a positive integer"
((DP_SIZE % DP_LOCAL_SIZE == 0)) || die "DP_SIZE must be divisible by DP_LOCAL_SIZE"
NODE_COUNT=$((DP_SIZE / DP_LOCAL_SIZE))
((NODE_RANK < NODE_COUNT)) || die "node rank ${NODE_RANK} is outside topology 0-$((NODE_COUNT - 1))"
DP_START_RANK=$((NODE_RANK * DP_LOCAL_SIZE))

LOCAL_IP="${LOCAL_IP:-${POD_IP:-${VLLM_HOST_IP:-}}}"
if [[ -z "${LOCAL_IP}" ]]; then
    LOCAL_IP="$(detect_local_ip)"
fi

if [[ "${NODE_RANK}" == "0" ]]; then
    DP_MASTER_ADDR="${DP_MASTER_ADDR:-${LOCAL_IP}}"
else
    [[ -n "${DP_MASTER_ADDR:-}" ]] || die "DP_MASTER_ADDR is required on decode node ${NODE_RANK}"
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    command -v "${VLLM_BIN}" >/dev/null 2>&1 || die "vllm command was not found: ${VLLM_BIN}"
    if [[ "${SKIP_MODEL_PATH_CHECK:-0}" != "1" ]]; then
        [[ -d "${MODEL_PATH}" ]] || die "model directory does not exist: ${MODEL_PATH}"
    fi
fi

export STARAGENT_DISABLED="${STARAGENT_DISABLED:-true}"
export ALIYUN_LOG_ENV_TAGS="${ALIYUN_LOG_ENV_TAGS:-MODEL_INSTANCE_NAME|MODEL_SERVICE_NAME|MODEL_NAME}"
export MODEL_PATH
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# This override must match the vLLM package actually installed in the image.
export VLLM_VERSION="${VLLM_VERSION:-0.20.2}"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="${OTEL_EXPORTER_OTLP_TRACES_PROTOCOL:-http/protobuf}"
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/}"
export ASCEND_PROCESS_LOG_PATH="${ASCEND_PROCESS_LOG_PATH:-/home/admin/logs/ascend/}"
export VLLM_LOGGING_CONFIG_PATH="${VLLM_LOGGING_CONFIG_PATH:-/home/admin/vllm/production_logging_config.json}"
export API_SERVER_COUNT="${API_SERVER_COUNT:-1}"
export PORT
export VLLM_BASE_PORT
export LOCAL_DP_SIZE="${LOCAL_DP_SIZE:-${DP_LOCAL_SIZE}}"
export NET_CARD_NAME="${NET_CARD_NAME:-${NETWORK_INTERFACE}}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-${LOCAL_IP}}"
export HCCL_IF_IP="${HCCL_IF_IP:-${LOCAL_IP}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE}}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${NETWORK_INTERFACE}}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE}}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-3600000}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-30000}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-204}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2560}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export VLLM_ASCEND_APPLY_DSV4_PATCH="${VLLM_ASCEND_APPLY_DSV4_PATCH:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export USE_MULTI_GROUPS_KV_CACHE="${USE_MULTI_GROUPS_KV_CACHE:-1}"
export USE_MULTI_BLOCK_POOL="${USE_MULTI_BLOCK_POOL:-1}"

if [[ "${ENABLE_ASYNC_MODEL_MOUNT}" == "1" ]]; then
    MODEL_MANAGER_ROOT="${MODEL_MANAGER_ROOT:-/a3_inference/itask/workdir/shared/ysl/async_model_mount_mock}"
    MOCK_MODEL_META_PATH="${MOCK_MODEL_META_PATH:-${BOOTSTRAP_MODEL:-${MODEL_PATH}}}"
    MOCK_MODEL_WEIGHT_PATH="${MOCK_MODEL_WEIGHT_PATH:-${FULL_MODEL:-}}"
    MOCK_MODEL_READY_FILE="${MOCK_MODEL_READY_FILE:-${READY_FILE:-}}"
    [[ -n "${MOCK_MODEL_WEIGHT_PATH}" ]] ||
        die "set FULL_MODEL or MOCK_MODEL_WEIGHT_PATH when ENABLE_ASYNC_MODEL_MOUNT=1"

    export PYTHONPATH="${MODEL_MANAGER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export VLLM_ASCEND_ASYNC_MODEL_MOUNT=1
    export MOCK_MODEL_META_PATH
    export MOCK_MODEL_WEIGHT_PATH
    export MOCK_MODEL_MOUNT_DELAY_SEC="${MOCK_MODEL_MOUNT_DELAY_SEC:-0}"
    export MOCK_MODEL_MOUNT_TIMEOUT_SEC="${MOCK_MODEL_MOUNT_TIMEOUT_SEC:-3600}"
    export MOCK_MODEL_MOUNT_POLL_INTERVAL_SEC="${MOCK_MODEL_MOUNT_POLL_INTERVAL_SEC:-1}"
    export MOCK_MODEL_VALIDATE_PATH="${MOCK_MODEL_VALIDATE_PATH:-1}"
    if [[ -n "${MOCK_MODEL_READY_FILE:-}" ]]; then
        export MOCK_MODEL_READY_FILE
    fi

    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        [[ -d "${MODEL_MANAGER_ROOT}/model_manager" ]] ||
            die "model_manager package does not exist under ${MODEL_MANAGER_ROOT}"
        python3 -c 'from model_manager.apis import get_model_path_from_manager' ||
            die "cannot import model_manager.apis; check MODEL_MANAGER_ROOT and PYTHONPATH"
    fi
else
    export VLLM_ASCEND_ASYNC_MODEL_MOUNT=0
fi

MODEL_LOADER_EXTRA_CONFIG="${MODEL_LOADER_EXTRA_CONFIG:-}"
if [[ -z "${MODEL_LOADER_EXTRA_CONFIG}" ]]; then
    MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load":true,"num_threads":128}'
fi

model_loader_args=()
if [[ "${ENABLE_RFORK}" == "1" ]]; then
    PLANNER_HOST="${PLANNER_HOST:-}"
    PLANNER_PORT="${PLANNER_PORT:-1223}"
    PLANNER_URL="${PLANNER_URL:-}"
    if [[ -z "${PLANNER_URL}" ]]; then
        [[ -n "${PLANNER_HOST}" ]] || die "set PLANNER_URL or PLANNER_HOST when ENABLE_RFORK=1"
        PLANNER_URL="http://${PLANNER_HOST}:${PLANNER_PORT}"
    fi
    if [[ "${ENABLE_ASYNC_MODEL_MOUNT}" == "1" ]]; then
        RFORK_MODEL_URL="${RFORK_MODEL_URL:-${FULL_MODEL:-${MOCK_MODEL_WEIGHT_PATH}}}"
    else
        RFORK_MODEL_URL="${RFORK_MODEL_URL:-${MODEL_PATH}}"
    fi
    MODEL_DEPLOY_STRATEGY_NAME="${MODEL_DEPLOY_STRATEGY_NAME:-dsv4-flash-a3-decode-dp16-tp1-rfork-v1}"
    RFORK_SEED_TIMEOUT_SEC="${RFORK_SEED_TIMEOUT_SEC:-30}"
    RFORK_CONFIG=$(printf \
        '{"model_url":"%s","model_deploy_strategy_name":"%s","rfork_scheduler_url":"%s","rfork_seed_timeout_sec":%s}' \
        "${RFORK_MODEL_URL}" "${MODEL_DEPLOY_STRATEGY_NAME}" "${PLANNER_URL}" "${RFORK_SEED_TIMEOUT_SEC}")
    model_loader_args=(--load-format rfork --model-loader-extra-config "${RFORK_CONFIG}")
    export VLLM_ASCEND_BALANCE_SCHEDULING="${VLLM_ASCEND_BALANCE_SCHEDULING:-0}"

    if [[ "${DRY_RUN:-0}" != "1" ]]; then
        python3 -c 'from yr.datasystem import TransferEngine' ||
            die "cannot import yr.datasystem.TransferEngine"
        curl --silent --show-error --output /dev/null --noproxy '*' \
            --connect-timeout 5 --max-time 10 "${PLANNER_URL}/get_seed" ||
            die "cannot reach RFork planner: ${PLANNER_URL}"
    fi
else
    model_loader_args=(--model-loader-extra-config "${MODEL_LOADER_EXTRA_CONFIG}")
fi

ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-}"
if [[ -z "${ADDITIONAL_CONFIG}" ]]; then
    ADDITIONAL_CONFIG='{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"recompute_scheduler_enable":true}'
fi

KV_TRANSFER_CONFIG=$(printf \
    '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"%s","engine_id":"%s","kv_connector_extra_config":{"prefill":{"dp_size":4,"tp_size":4},"decode":{"dp_size":16,"tp_size":1}}}' \
    "${KV_PORT}" "${ENGINE_ID}")

OTLP_TRACES_ENDPOINT="${OTLP_TRACES_ENDPOINT-https://antcollector.alipay.com/namespace/aicloud/task/otlptrace/otlp/api/v1/traces}"

vllm_args=(
    "${VLLM_BIN}" serve "${MODEL_PATH}"
    --host 0.0.0.0
    --port "${PORT}"
    --trust-remote-code
    --served-model-name "${SERVED_MODEL_NAME}"
    --distributed-executor-backend mp
    "${model_loader_args[@]}"
    --enable-log-requests
    --enable-prompt-tokens-details
    --token-level-profiling
    --data-parallel-size "${DP_SIZE}"
    --data-parallel-size-local "${DP_LOCAL_SIZE}"
    --data-parallel-start-rank "${DP_START_RANK}"
    --data-parallel-address "${DP_MASTER_ADDR}"
    --data-parallel-rpc-port "${DP_RPC_PORT}"
    --tensor-parallel-size 1
    --enable-expert-parallel
    --seed 1024
    --max-model-len 1048576
    --max-num-batched-tokens 60
    --max-num-seqs 30
    --no-disable-hybrid-kv-cache-manager
    --no-enable-prefix-caching
    --safetensors-load-strategy prefetch
    --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}'
    --block-size 128
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --gpu-memory-utilization 0.9
    --quantization ascend
    --async-scheduling
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --additional-config "${ADDITIONAL_CONFIG}"
    --kv-transfer-config "${KV_TRANSFER_CONFIG}"
)

if [[ -n "${OTLP_TRACES_ENDPOINT}" ]]; then
    vllm_args+=(--otlp-traces-endpoint "${OTLP_TRACES_ENDPOINT}")
fi

if [[ "${NODE_RANK}" != "0" ]]; then
    vllm_args+=(--headless)
fi

vllm_args+=("$@")

echo "Starting DeepSeek V4 Flash A3 decode node ${NODE_RANK}/${NODE_COUNT}"
echo "  model:       ${MODEL_PATH}"
echo "  local IP:    ${LOCAL_IP}"
echo "  DP master:   ${DP_MASTER_ADDR}:${DP_RPC_PORT}"
echo "  DP ranks:    ${DP_START_RANK}-$((DP_START_RANK + DP_LOCAL_SIZE - 1))"
echo "  parallelism: DP${DP_SIZE} / local DP${DP_LOCAL_SIZE} / TP1"
echo "  KV role:     consumer, port ${KV_PORT}, engine ${ENGINE_ID}"
if [[ "${ENABLE_RFORK}" == "1" ]]; then
    echo "  RFork:       enabled, planner ${PLANNER_URL}"
    echo "  strategy:    ${MODEL_DEPLOY_STRATEGY_NAME}"
    echo "  model URL:   ${RFORK_MODEL_URL}"
    if [[ "${ENABLE_ASYNC_MODEL_MOUNT}" == "1" ]]; then
        echo "  async mount: enabled"
        echo "  bootstrap:   ${MOCK_MODEL_META_PATH}"
        echo "  full model:  ${MOCK_MODEL_WEIGHT_PATH}"
        if [[ -n "${MOCK_MODEL_READY_FILE:-}" ]]; then
            echo "  ready file:  ${MOCK_MODEL_READY_FILE}"
        else
            echo "  mount delay: ${MOCK_MODEL_MOUNT_DELAY_SEC}s"
        fi
    else
        echo "  async mount: disabled"
    fi
else
    echo "  RFork:       disabled"
fi

run_command "${vllm_args[@]}"
