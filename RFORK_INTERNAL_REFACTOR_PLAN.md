# RFork Internal Architecture Refactor Plan

## Scope

This refactor is limited to `vllm_ascend/model_loader/rfork`, its example
planner, and RFork tests. It does not modify, import from, or create shared
code with NetLoader.

The refactor preserves the current RFork behavior:

- planner-based seed discovery and leases;
- protocol-v2 compatibility fingerprints;
- checkpoint-layout and processed-layout transfers;
- YuanRong logical/backing memory registration;
- seed health checks, heartbeat reporting, and explicit shutdown;
- fallback to the default vLLM model loader;
- quantized, draft, TP, PP, and EP identity handling.

## Problems to Solve

1. `RForkModelLoader` and `RForkWorker` both orchestrate lifecycle cleanup.
2. `transfer_backend.py` owns tensor discovery, manifest parsing, HTTP metadata
   requests, allocator registration, native transfer, and engine lifecycle.
3. Configuration is parsed through nested functions inside the loader and then
   expanded into a long worker constructor.
4. Seed leases and transfer metadata cross module boundaries as untyped
   dictionaries and tuples.
5. Seed-server startup uses a handle-or-integer return contract and retains
   numeric compatibility methods.
6. Failure handling is expressed through loosely related booleans rather than
   one explicit lifecycle state.

## Target Module Layout

```text
vllm_ascend/model_loader/rfork/
├── __init__.py
├── config.py             # validated RForkConfig resolution
├── types.py              # typed leases, seed metadata, and lifecycle state
├── manifest.py           # RFork-only tensor discovery and manifest helpers
├── seed_client.py        # seed metadata HTTP client
├── planner_client.py     # planner lease and heartbeat client
├── seed_server.py        # owned seed HTTP service
├── transfer_backend.py   # YuanRong engine and memory/transfer operations
├── session.py            # sole lifecycle owner
└── rfork_loader.py       # model initialization, fallback, and high-level flow
```

## Ownership Rules

### `RForkModelLoader`

- Selects RFork versus default loading.
- Initializes and post-processes the model.
- Delegates all RFork resource transitions to `RForkSession`.
- Does not call planner, seed server, or transfer backend cleanup primitives
  individually.

### `RForkSession`

- Is the sole owner of the planner lease, TransferEngine backend, seed server,
  and heartbeat thread.
- Enforces explicit lifecycle transitions.
- Provides high-level operations for loading from a seed, preparing a fallback
  model to become a seed, stopping publication, and final shutdown.
- Retains memory owners whenever server or TransferEngine shutdown is
  incomplete.

### `RForkTransferBackend`

- Owns one YuanRong TransferEngine instance.
- Registers, transfers, unregisters, and finalizes native resources.
- Uses manifest and seed-client helpers but does not make planner decisions or
  start HTTP services.

### Planner and seed services

- `RForkPlannerClient` owns planner HTTP requests and typed lease/report state.
- `seed_client.py` owns seed metadata retrieval.
- `seed_server.py` owns one HTTP server handle with a single concrete startup
  result type.

## Lifecycle State Model

```text
INITIALIZED
   ├─ acquire seed ─> LEASED
   ├─ register local memory ─> REGISTERED
   ├─ transfer complete/release lease ─> REGISTERED
   ├─ publish healthy seed ─> SERVING
   ├─ stop publication ─> REGISTERED
   ├─ unregister for fallback retry ─> INITIALIZED
   └─ finalize ─> FINALIZED
```

Invalid transitions fail without discarding retry state. A live seed server
always pins registered tensor owners. A failed TransferEngine Finalize keeps
the session non-finalized and retryable.

## Typed Boundaries

- `SeedLease`: planner allocation identity and endpoint.
- `SeedAdvertisement`: the endpoint last accepted by the planner.
- `SeedTransferInfo`: session id plus weight and shape manifests.
- `RForkLifecycleState`: explicit resource state.
- `RForkConfig`: resolved user/environment configuration.

Wire serialization remains isolated at HTTP boundaries. Internal code does not
index arbitrary dictionaries for lease or transfer endpoint fields.

## Compatibility Policy

- No compatibility fallback to YuanRong TransferEngine versions lacking
  `MemoryRegistration`, `batch_register_memory_ex`, or `finalize`.
- Internal handle-or-port and old worker-return compatibility paths are removed.
- Existing RFork protocol-v2 seed identity and current planner endpoints remain
  unchanged by this structural refactor.

## Implementation Phases

1. Add typed config and boundary objects.
2. Move planner behavior into `RForkPlannerClient`.
3. Replace `RForkWorker` with `RForkSession` and move all cleanup decisions into
   the session.
4. Simplify loader orchestration to high-level session calls.
5. Extract RFork-only tensor/manifest helpers and seed metadata HTTP requests
   from `transfer_backend.py`.
6. Make seed-server startup return only `RForkSeedServerHandle` or raise.
7. Update tests around typed objects and lifecycle transitions.

## Verification

- Run every RFork unit test.
- Cover valid and invalid lifecycle transitions.
- Cover planner lease retention and release retries.
- Cover seed startup/shutdown failures retaining registered memory.
- Cover TransferEngine Finalize retry and owner retention.
- Run `ruff check`, `ruff format --check`, `compileall`, and `git diff --check`.
- Report NPU/HiXL end-to-end validation as a separate hardware requirement.

## Done Criteria

- Loader contains no primitive stop/release/unregister/finalize orchestration.
- Session is the sole owner of RFork runtime resources.
- Seed leases and transfer metadata are typed internally.
- `transfer_backend.py` no longer contains planner or seed HTTP parsing helpers.
- Seed-server startup has one success type and one failure mechanism.
- No NetLoader file is changed.
- Existing RFork behavior and regression coverage remain intact.
