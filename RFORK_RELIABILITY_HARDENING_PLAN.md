# RFork Reliability Hardening Plan

## 1. Goal

This change hardens RFork from a best-effort warm-start path into a fail-safe
weight-transfer path. A failed or incompatible transfer must either fall back
to the default loader or return a fully validated model. It must never leave a
live planner lease, an untracked registered memory region, or a seed service
that advertises stale tensor addresses.

## 2. Scope

The implementation covers the following areas:

1. Bound every planner and seed HTTP request with a validated finite timeout.
2. Make seed HTTP server, heartbeat, lease, and registered-memory lifecycles
   explicit and idempotent.
3. Preserve registered tensor owners until memory unregister succeeds, including
   unregister-failure and retry paths.
4. Select checkpoint-layout versus processed-layout transfer from the actual
   Ascend post-load conversion requirements, including unquantized NZ weights.
5. Upgrade the compatibility protocol:
   - canonical compatibility fingerprint;
   - collision-free opaque seed keys;
   - dtype, shape, tensor count, and total-byte validation;
   - strict local/remote manifest equality;
   - validated native pointers and bounded transfer chunks.
6. Add shared-token authentication hooks and configurable bind/advertise
   addresses for trusted-cluster deployments. Transport encryption remains a
   deployment responsibility and HTTPS planner URLs remain supported.
7. Add unit coverage for the loader, transfer backend, worker, seed protocol,
   seed server, and mock planner state machines.

## 3. Design

### 3.1 Configuration

`model_loader_extra_config` remains the primary configuration source. Each
field accepts an environment variable fallback read directly by the loader
(`MODEL_URL`, `MODEL_DEPLOY_STRATEGY_NAME`, `RFORK_SCHEDULER_URL`,
`RFORK_SEED_TIMEOUT_SEC`, `RFORK_REQUEST_TIMEOUT_SEC`, `RFORK_AUTH_TOKEN`,
`RFORK_SEED_BIND_HOST`, `RFORK_SEED_ADVERTISE_HOST`).

Numeric configuration rejects booleans, NaN, infinity, and non-positive values.
The auth token is sensitive and must never be logged.

### 3.2 Compatibility identity

RFork hashes a canonical JSON descriptor containing model identity,
deployment strategy, dtype, quantization method plus a digest of the full
quantization config, revision, model architecture, parallel world sizes, and
Ascend weight layout mode. The planner sees an opaque seed key, so separators
inside user-controlled values cannot create collisions.

The transfer manifest additionally carries per-tensor dtype, logical shape,
element count, element size, tensor count, and total bytes. The receiver rejects
missing, additional, empty, malformed, or incompatible manifests before native
transfer starts.

### 3.3 Transfer and memory state

Memory registration is transactional:

- a second registration is rejected while blocks remain registered;
- materialized CPU parameters/buffers reject RFork so mixed-device state cannot
  be silently omitted;
- every transferable tensor must be covered by a registered allocator block;
- zero-tensor manifests are rejected;
- unregister failure retains block tracking and tensor owners for retry;
- no caller may overwrite a failed unregister state.

Transfers are split so every native call is at most 1 GiB and 512 pointer
segments, including a single tensor larger than 1 GiB.

### 3.4 Seed lifecycle

The seed server returns an explicit handle with `stop()`. `RForkWorker` owns the
server handle, heartbeat stop event, and heartbeat thread. Shutdown order is:

1. stop advertising and deregister the seed on a best-effort basis;
2. stop heartbeat;
3. stop and join the HTTP server;
4. unregister memory;
5. release tensor owners.

An instance is advertised only after transfer, post-load processing, and
`model.eval()` finish. A seed startup failure does not reload an already valid
model; it cleans registration state and returns the model without seed service.
Worker shutdown is idempotent and registered for process exit.

### 3.5 Planner leases

Lease release results propagate to callers and use bounded retries. The mock
planner independently expires abandoned leases, so a live seed cannot remain at
capacity forever after a receiver crash or network failure.

### 3.6 Fallback

Worker construction and TransferEngine initialization are inside the guarded
RFork path. Missing dependencies, invalid RFork configuration, group-state
errors, transfer failures, and manifest incompatibility all fall back to the
default loader. If no worker was created, the fallback model is returned without
starting a seed service.

## 4. Tests

Required unit coverage:

- finite timeout propagation and timeout failure;
- manifest dtype/count/byte/name mismatch;
- strict chunk limits for tensors larger than 1 GiB;
- unregister failure followed by retry/re-registration;
- server start/stop and health timeout cleanup;
- heartbeat stop and lease-release propagation;
- fallback when worker construction fails;
- unquantized NZ processed-layout selection;
- seed-key collision resistance and fingerprint changes;
- orphan lease expiration in the mock planner;
- authentication success and rejection paths.

Required NPU validation after unit tests:

- BF16 with `weight_nz_mode=0` and `2`;
- one supported quantized model;
- one MTP draft model;
- forced transfer failure followed by fallback and KV-cache allocation;
- forced unregister and seed-server failures;
- TP/PP/EP multi-rank seed isolation.

## 5. Acceptance Criteria

- Formatting, lint, and all RFork unit tests pass.
- No HTTP call in RFork is unbounded.
- No failure path can overwrite or forget a registered memory region.
- No seed is advertised before the model is finalized.
- A stopped or failed seed service cannot continue heartbeating.
- Incompatible manifests fail before native memory transfer.
- Default-loader fallback remains usable when RFork initialization fails.
- NPU-only validation requirements are reported explicitly when the local
  environment cannot execute them.
