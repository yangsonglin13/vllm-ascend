# RFork Guide

This guide explains how to use **RFork** as a model-loader plugin in **vLLM Ascend**.

---

## Overview

RFork is a warm-start weight loading path for vLLM Ascend. Instead of always reading model weights from storage, a new instance can request a compatible **seed** instance from an external planner, then pull weights directly from that seed through `YuanRong TransferEngine`.

The RFork loading flow in the current implementation is:

1. vLLM starts with `--load-format rfork`.
2. RFork builds a **seed key** from the model identity and deployment topology.
3. RFork asks the planner for an available seed matching that key.
4. If a seed is returned, the new instance initializes the model structure on its local NPU, registers local weight memory, fetches the remote transfer-engine metadata from the seed, and performs batch weight transfer into local parameter buffers.
5. If no seed is available, or any transfer step fails, RFork cleans up and falls back to the default loader.
6. RFork completes post-load processing and switches the model to evaluation mode before it starts a local seed service and advertises it to the planner. A seed-service startup failure does not reload the valid model; RFork only cleans up its registered memory.

## Flowchart

![rfork flowchart](./images/rfork_flowchart.jpg)

## Application Scenarios

- **Scale-out after a first successful load**: The first instance may still load from storage, but later instances with the same deployment identity can reuse it as a seed and shorten startup time.
- **Elastic serving clusters**: Because RFork asks a planner for available seeds, it fits clusters where instances are created and reclaimed dynamically.
- **Topology-sensitive deployments**: RFork encodes optional `pp_rank`, `tp_rank`, and optional `ep_rank` into the seed key, so only shard-compatible instances are matched together. The `kv_role` and the physical node are deliberately excluded, so prefill and decode instances on different nodes can share seeds.

---

## Usage

To enable RFork, pass `--load-format rfork` and provide RFork settings through `--model-loader-extra-config` as a JSON string.

### RFork Prerequisites

- Install the runtime dependency `YuanRong TransferEngine` on every RFork instance.
- Run a planner service that implements the RFork seed protocol. A simple mock planner script is provided at [`rfork_planner.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rfork/rfork_planner.py).

### Configuration Fields

| Field Name | Type | Description | Allowed Values / Notes |
|------------|------|-------------|------------------------|
| **model_url** | String | Logical model identifier used to build the RFork seed key. | Required for RFork transfer. Instances that should share seeds must use the same value. |
| **model_deploy_strategy_name** | String | Deployment strategy identifier used together with `model_url` to build the seed key. | Required for RFork transfer. Instances that should share seeds must use the same value. |
| **rfork_scheduler_url** | String | Base URL of the planner service used for seed allocation, release, and heartbeat. | Required for planner-based matching. Example: `http://127.0.0.1:1223`. |
| **rfork_seed_timeout_sec** | Number | Timeout for waiting until the local seed HTTP service becomes healthy after startup. | Optional. Default: `5.0`. Must be greater than `0`. Invalid values fall back to the default. |
| **rfork_request_timeout_sec** | Number | Timeout for every planner and seed HTTP request. | Optional. Default: `10.0`. Must be finite and greater than `0`; invalid values fall back to the default. |
| **rfork_seed_bind_host** | String | Local address/interface for the seed HTTP server. | Optional. Default: `0.0.0.0`. |
| **rfork_seed_advertise_host** | String | Address reported to the planner for later receivers. | Optional. Default: auto-detect the local address. |

Each field also accepts an environment variable fallback:

| Environment variable | Default | Notes |
|----------------------|---------|-------|
| `MODEL_URL` | unset | |
| `MODEL_DEPLOY_STRATEGY_NAME` | unset | |
| `RFORK_SCHEDULER_URL` | unset | |
| `RFORK_SEED_TIMEOUT_SEC` | `5.0` | Finite and positive only. |
| `RFORK_REQUEST_TIMEOUT_SEC` | `10.0` | Finite and positive only. |
| `RFORK_SEED_BIND_HOST` | `0.0.0.0` | |
| `RFORK_SEED_ADVERTISE_HOST` | auto-detect | |

`model_loader_extra_config` takes precedence over environment values. Numeric
values reject booleans, NaN, infinity, and non-positive values.

### How RFork Matches Seeds

RFork does not match instances by `model_url` alone. The seed identity is composed from:

- `model_url`
- `model_deploy_strategy_name`
- `pp_rank` when pipeline parallel size is greater than 1
- `tp_rank`
- `ep_rank` when expert parallelism is enabled for an MoE model
- `draft` role when the worker runs as a draft model

The parallel ranks keep seed and receiver on the same model shard, while the
disaggregation role (`kv_role`) and the physical node are deliberately excluded
so prefill and decode instances on different nodes can share seeds.

The identity is anchored by a SHA256 compatibility fingerprint covering model
revision, dtype, quantization method, a digest of the full quantization config,
model architecture, tensor/pipeline/expert parallel world sizes, Ascend NZ
layout mode, and hardware layout policy. This prevents an old or differently
configured seed from being selected merely because its textual key happens to
match.

Two instances must agree on model identity and parallel layout before the planner will treat them as interchangeable seeds. The seed key is an opaque SHA256 digest, so seed and receiver instances must run the same RFork protocol to derive the same key.

### Quantized Models

For quantized models, RFork transfers tensors after Ascend weight post-processing instead of raw checkpoint parameters. The receiver first builds the same post-load tensor layout as the seed, then RFork copies the live NPU tensors used by inference.

This path handles Ascend quantization changes such as weight transposition, NZ format conversion, packed weights, derived scale tensors, and MLA/SFA runtime tensors such as `W_UV` and `W_UK_T`. Empty tensors that were released during post-processing are not included in the transfer manifest.

When validating RFork for a quantized model:

- Apply the same vLLM Ascend code to both the seed instance and the receiver instance.
- Restart the planner and all vLLM instances after changing RFork code, because existing seeds keep their old transfer metadata.
- Use a new `model_deploy_strategy_name` after changing model arguments or RFork code when operating with an older planner; compatibility fingerprints also reject incompatible old seeds before native transfer.
- A successful RFork transfer logs `transfer weights starts` and `transfer weights time`. The fallback path logs `RFork transfer failed`.

## Tested Models

The following table records models that have been explicitly tested with RFork weight transfer. A model should be added here only after RFork transfer succeeds and the loaded instance passes basic inference validation.

| Model | Precision / Quantization | Hardware | Validation Status | Notes |
|-------|--------------------------|----------|-------------------|-------|
| Qwen2.5-7B | BF16 | A2 | Tested | RFork transfer has been validated. |
| Qwen3-32B | BF16 | A2 | Tested | RFork transfer has been validated. |
| Qwen3-235B-A22B | BF16 | A2 | Tested | RFork transfer has been validated. |
| DeepSeek-V4-Flash-W8A8-MTP | W8A8 | A2 | Tested | RFork transfer with MTP draft model has been validated. |
| GLM5-W4A8 | W4A8 | A2 | Tested | RFork transfer has been validated. |
| Kimi2.5-W4A8 | W4A8 | A2 | Tested | RFork transfer has been validated. |

---

## Example Commands & Placeholders

> Replace parts in `<...>` before running.

### 1. Install YuanRong TransferEngine

```shell
pip install openyuanrong-transfer-engine
```

### 2. Start the Planner

A simple planner implementation is provided at [`rfork_planner.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rfork/rfork_planner.py).

```shell
python rfork_planner.py \
  --host 0.0.0.0 \
  --port <planner_port>
```

### 3. Start vLLM Instances

Use the same RFork startup command for both the first instance and later instances in the same deployment.

For the first instance, the planner usually has no compatible seed yet, so RFork falls back to the default loader. After loading finishes, that instance starts its local seed service and reports itself to the planner.

For later instances, if the planner can allocate a compatible seed, RFork will try to transfer weights from the existing seed instance before falling back to the default loader.

```shell
export RFORK_CONFIG='{
  "model_url": "<model_url>",
  "model_deploy_strategy_name": "<deploy_strategy>",
  "rfork_scheduler_url": "http://<planner_ip>:<planner_port>",
  "rfork_request_timeout_sec": 10.0,
  "rfork_seed_bind_host": "0.0.0.0",
  "rfork_seed_advertise_host": "<seed_ip>"
}'

vllm serve <model_path> \
  --tensor-parallel-size 1 \
  --served-model-name <served_model_name> \
  --port <port> \
  --load-format rfork \
  --model-loader-extra-config "${RFORK_CONFIG}"
```

### Placeholder Descriptions

- `<model_path>`: Model path or model identifier passed to `vllm serve`.
- `<served_model_name>`: Service name exposed by vLLM.
- `<planner_ip>`: IP address or hostname of the RFork planner.
- `<planner_port>`: Listening port of the RFork planner.
- `<model_url>`: Stable model identity string used to build the RFork seed key.
- `<deploy_strategy>`: Stable deployment-strategy name used to build the RFork seed key.
- `<port>`: Serving port of the vLLM instance being started.

---

## Note & Caveats

- RFork requires `YuanRong TransferEngine` at runtime. If the package is missing or cannot be initialized, the default loader is used and no seed service is started by that RFork worker.
- If RFork is used, **each worker process** must bind a listening port. The seed bind host is configurable and the port is assigned randomly.
- RFork currently requires all materialized model parameters and registered buffers to reside on NPU. Mixed CPU/NPU or CPU-offloaded model state is rejected and falls back to the default loader rather than performing a partial transfer.
- Keep the seed bind/advertise addresses reachable from the receiver workers. Transport encryption and access control remain the deployment's responsibility, so use HTTPS and network isolation for untrusted networks.
- RFork weight transfer does not support dynamic EPLB because expert weights and placement can change after the seed service starts. If `eplb_config.dynamic_eplb` or `eplb_config.expert_map_record_path` enables dynamic EPLB, RFork transfer is bypassed and the model is loaded through the default model loader.
- The example [`rfork_planner.py`](https://github.com/vllm-project/vllm-ascend/blob/main/examples/rfork/rfork_planner.py) is only a simple mock implementation. If you need stronger scheduling, capacity management, or production-grade availability behavior, implement your own planner based on the RFork seed protocol.
