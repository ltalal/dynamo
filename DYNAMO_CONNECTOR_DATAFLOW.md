# DynamoConnector Data Flow Analysis

## Initialization

### Leader (Scheduler) Initialization

- **[dynamo_connector.py:42](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L42) `DynamoConnector.__init__(vllm_config, role='SCHEDULER')`**
  - Creates leader instance when role is SCHEDULER
  - **[connector_leader.py:35](lib/kvbm/python/kvbm/vllm_integration/connector_leader.py#L35) `KvConnectorLeader.__init__(vllm_config, engine_id)`**
    - **[connector_leader.py:69](lib/kvbm/python/kvbm/vllm_integration/connector_leader.py#L69) `KvbmLeader(world_size, drt)`**
      - Initializes ZMQ context for worker communication
      - Creates transfer coordinator instance
    - **[connector/leader.rs:113](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L113) `PyKvConnectorLeader::new(worker_id, page_size, leader)`**
      - **[connector/leader.rs:138](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L138) `ConnectorSlotManager::new(block_manager, leader, metrics)`**
        - Creates slot manager to track request states
        - Initializes block manager pools (device, host, disk)
        - Sets up metrics collectors
      - **[connector/leader.rs:147](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L147) `tokio::spawn(async move { engine.run().await })`**
        - Spawns LocalTransferEngine as async task
        - Engine listens for onboard/offload requests
        - Coordinates with KvbmLeader for actual transfers

### Worker (Executor) Initialization

- **[dynamo_connector.py:42](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L42) `DynamoConnector.__init__(vllm_config, role='WORKER')`**
  - Creates worker instance when role is WORKER
  - **[connector_worker.py:30](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L30) `KvConnectorWorker.__init__(vllm_config, engine_id)`**
    - Lazy initialization - worker not fully created until first use
    - **[connector/worker.rs:89](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L89) `PyKvConnectorWorker::new(drt, worker_id)`**
      - **[connector/scheduler.rs:45](lib/llm/src/block_manager/connector/scheduler.rs#L45) `Scheduler::new()`**
        - Returns tuple: `(Scheduler, WorkerSchedulerClient, TransferSchedulerClient)`
        - `WorkerSchedulerClient`: Worker-side request tracking
        - `TransferSchedulerClient`: Transfer engine coordination
        - `Scheduler`: Event loop for matching requests
      - **[connector/worker.rs:95](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L95) `tokio::spawn(async move { scheduler.run().await })`**
        - Spawns scheduler task to coordinate transfers
        - Matches worker requests with leader requests
        - Executes two-phase commit for transfers
    - **[connector_worker.py:63](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L63) `register_kv_caches(kv_caches)`** (called separately)
      - **[connector_worker.py:67](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L67) `KvbmWorker.new(config)`**
        - Connects to leader via ZMQ
        - Registers tensor buffers for GPU operations

---

## Request Processing Flow (Leader Side)

### 1. Cache Lookup and Prefetch Decision

- **[dynamo_connector.py:70](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L70) `get_num_new_matched_tokens(request, num_computed_tokens)`**
  - vLLM calls this to check if prefix is cached
  - **[dynamo_connector.py:72](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L72) `_create_slot(request)`** (if slot doesn't exist)
    - Creates new slot for request
    - **[connector_leader.py:82](lib/kvbm/python/kvbm/vllm_integration/connector_leader.py#L82) `connector.create_slot(kvbm_request, tokens)`**
      - **[connector/leader.rs:169](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L169) `PyKvConnectorLeader::create_slot(request_id, tokens, salt_hash)`**
        - **[connector/vllm_connector_slot.rs:82](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L82) `VllmConnectorSlot::new(request_id, tokens, salt_hash)`**
          - Initializes slot state machine in `Initialized` state
          - Creates empty pending_operations queue
          - Sets up token tracking
  - **[connector_leader.py:96](lib/kvbm/python/kvbm/vllm_integration/connector_leader.py#L96) `connector.get_num_new_matched_tokens()`**
    - **[connector/vllm_connector_slot.rs:139](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L139) `acquire_local_matches(num_computed_tokens)`**
      - **[connector/vllm_connector_slot.rs:152](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L152) `lookup_local_blocks_for_tokens()`**
        - Searches host pool for matching token blocks
        - Searches disk pool for matching token blocks
        - Returns `(matched_token_blocks, matched_block_ids)`
      - **If matches found:**
        - **[connector/vllm_connector_slot.rs:190](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L190) `stage_blocks_for_onboarding(num_external_tokens)`**
          - Marks blocks for loading from host/disk
          - Creates staging metadata for transfer
          - Sets `self.state = OnboardStaged(num_external_tokens)`
        - Returns `(num_external_tokens, true)` to indicate async operation
      - **If no matches:**
        - Returns `(0, false)` - no cached data available

### 2. Block Allocation and Onboarding

- **[dynamo_connector.py:79](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L79) `update_state_after_alloc(request, blocks, num_external_tokens)`**
  - vLLM calls after allocating GPU blocks for the request
  - **[dynamo_connector.py:86](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L86) `slot.append_mutable_device_blocks(block_ids)`**
    - **[connector/vllm_connector_slot.rs:240](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L240) `append_mutable_device_blocks(block_ids)`**
      - Stores device block IDs assigned by vLLM
      - Maps position in sequence to GPU block ID
  - **[dynamo_connector.py:88](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L88) `slot.trigger_onboarding(num_external_tokens)`**
    - **[connector/vllm_connector_slot.rs:266](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L266) `trigger_onboarding(num_external_tokens)`**
      - **[connector/vllm_connector_slot.rs:281](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L281) `create_onboard_requests()`**
        - For each staged block:
          - Creates `LocalOnboardRequest` with:
            - `src_blocks`: host/disk block IDs
            - `dst_blocks`: device block IDs
            - `transfer_type`: TransferType::Load
        - Appends to `self.pending_operations`
      - **[connector/leader.rs:225](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L225) `send_to_transfer_engine(operation)`**
        - Sends `LocalOnboardRequest` to LocalTransferEngine
        - Engine receives via mpsc channel
      - Sets `self.state = Onboarding(num_external_tokens)`
      - vLLM waits for completion before running forward pass

### 3. Metadata Broadcast to Workers

- **[dynamo_connector.py:94](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L94) `build_connector_metadata(scheduler_output)`**
  - vLLM calls before each iteration to sync workers
  - **For each new_request in scheduler_output:**
    - **[connector_leader.py:113](lib/kvbm/python/kvbm/vllm_integration/connector_leader.py#L113) `slot.apply_scheduler_output(scheduler_output, iteration)`**
      - **[connector/vllm_connector_slot.rs:330](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L330) `apply_scheduler_output()`**
        - **[connector/vllm_connector_slot.rs:365](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L365) `evaluate_offload_policy()`**
          - Checks if request should offload KV cache
          - Policy: offload if not scheduled for next iteration
          - Returns `should_offload: bool`
        - **If should_offload:**
          - **[connector/vllm_connector_slot.rs:410](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L410) `offload_blocks(block_ids, token_blocks)`**
            - Creates `LocalOffloadRequest` with:
              - `src_blocks`: device block IDs
              - `token_blocks`: semantic content metadata
              - `transfer_type`: TransferType::Store
            - **[connector/leader.rs:240](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L240) `send_to_transfer_engine(operation)`**
              - Sends to LocalTransferEngine
              - Engine allocates blocks in host/disk pools
              - Creates `LeaderTransferRequest` for workers
            - Appends operation to `self.pending_operations`
        - Updates slot state based on scheduler decision:
          - `Prefilling` / `Decoding` if scheduled
          - `SkippedPrefill` / `SkippedDecode` if not scheduled
  - **For each cached_request in scheduler_output:**
    - Same process as new_request (may trigger offload)
  - **[connector_leader.py:128](lib/kvbm/python/kvbm/vllm_integration/connector_leader.py#L128) `collect_pending_operations()`**
    - **[connector/vllm_connector_slot.rs:488](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L488) `take_pending_operations()`**
      - Drains `self.pending_operations` queue
      - Returns all transfer operations for this iteration
  - **[connector/mod.rs:89](lib/kvbm/src/block_manager/vllm/connector/mod.rs#L89) `ConnectorMetadata` serialization**
    - Serializes into bytes using bincode:
      - `new_slots`: List of new request IDs
      - `transfer_operations`: All pending Load/Store operations
      - `slot_state_updates`: State changes for tracking
    - Returns serialized bytes to vLLM
    - vLLM broadcasts to all workers via RPC

### 4. Request Completion

- **[dynamo_connector.py:106](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L106) `request_finished(request, block_ids)`**
  - vLLM calls when request generation is complete
  - **[connector_leader.py:148](lib/kvbm/python/kvbm/vllm_integration/connector_leader.py#L148) `slot.mark_as_finished(iteration)`**
    - **[connector/vllm_connector_slot.rs:520](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L520) `mark_as_finished()`**
      - **If pending_operations is not empty:**
        - Sets `self.state = Finishing`
        - Returns `(true, None)` - has async cleanup
        - Leader waits for transfers to complete
      - **Else:**
        - Sets `self.state = Finished`
        - Returns `(false, None)` - safe to free immediately
  - vLLM waits if has_pending_ops before freeing GPU blocks

---

## Request Processing Flow (Worker Side)

### 1. KV Cache Registration

- **[dynamo_connector.py:110](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L110) `register_kv_caches(kv_caches)`**
  - Called once during worker initialization
  - `kv_caches`: dict mapping layer names to GPU tensors
  - **[connector_worker.py:63](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L63) `connector.register_kv_caches(kv_caches)`**
    - **[connector_worker.py:67](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L67) `order_layers_by_index(kv_caches)`**
      - Sorts layers by numeric index (e.g., "0", "1", ..., "31")
      - Ensures consistent processing order
    - **[connector_worker.py:73](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L73) `create_cuda_events()`**
      - Creates CUDA event for each layer
      - Used to track GPU completion before offload
    - **[connector/worker.rs:110](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L110) `PyKvConnectorWorker::register_kv_caches()`**
      - **[connector_worker.py:88](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L88) `KvbmWorker.new(config)`** (lazy init)
        - Connects to leader via ZMQ
        - Establishes transfer coordination channel

### 2. Metadata Binding

- **[dynamo_connector.py:115](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L115) `bind_connector_metadata(metadata_bytes)`**
  - Called at start of each iteration with leader's instructions
  - **[connector/mod.rs:89](lib/kvbm/src/block_manager/vllm/connector/mod.rs#L89) `ConnectorMetadata` deserialization**
    - Deserializes bytes into:
      - `new_slots`: New requests to create
      - `transfer_operations`: Load/Store operations to execute
      - `slot_state_updates`: State synchronization
  - **[connector_worker.py:95](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L95) `connector.start_next_iteration()`**
    - **[connector/worker.rs:145](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L145) `start_next_iteration()`**
      - Clears `is_complete` flags from previous iteration
      - Resets `maybe_finished_onboarding` set
      - Prepares for new round of operations
  - **For each new_slot in metadata.new_slots:**
    - **[connector_worker.py:99](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L99) `connector.create_slot(request_id)`**
      - **[connector/worker.rs:163](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L163) `create_slot(request_id)`**
        - Creates worker-side slot tracking
        - Initializes `AtomicU64` completion counter
        - Stored in `self.slots` HashMap
  - **For each operation in metadata.transfer_operations:**
    - **If operation.transfer_type == TransferType::Load:**
      - **Onboarding operation - execute immediately**
      - **[connector_worker.py:108](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L108) `connector.enqueue_operation(operation)`**
        - **[connector/worker.rs:185](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L185) `enqueue_load_operation()`**
          - **[connector/scheduler.rs:210](lib/llm/src/block_manager/connector/scheduler.rs#L210) `WorkerSchedulerClient::submit_worker_request()`**
            - Creates `WorkerTransferRequest`:
              - `uuid`: Matches leader's request UUID
              - `request_id`: Request identifier
              - `src_blocks`: Host/disk block IDs
              - `dst_blocks`: Device block IDs
              - `transfer_type`: Load
              - `completion_handle`: Shared counter reference
            - Sends to Scheduler via mpsc channel
            - Scheduler waits for matching `LeaderTransferRequest`
          - Adds request_id to `self.maybe_finished_onboarding`
          - Worker will poll for completion before forward pass
    - **If operation.transfer_type == TransferType::Store:**
      - **Offloading operation - delay until GPU complete**
      - **[connector_worker.py:113](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L113) `store_offload_operation(operation)`**
        - Adds to `self.offloading_operations` buffer
        - Will enqueue after last layer GPU sync

### 3. Layer Execution and GPU Synchronization

- **[dynamo_connector.py:120](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L120) `start_load_kv(forward_context)`**
  - Called before each layer's forward pass
  - No-op in current implementation
  - Could be used for prefetch preparation

- **[dynamo_connector.py:125](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L125) `save_kv_layer(layer_name, kv_layer, attn_metadata)`**
  - Called after each layer's forward pass completes
  - `kv_layer`: GPU tensor with computed K/V values
  - **[connector_worker.py:130](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L130) `record_cuda_event(layer_name)`**
    - Records CUDA event on current stream
    - Ensures GPU work is complete before transfer
    - Event stored in `self.layer_events[layer_name]`
  - **[connector_worker.py:135](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L135) `self.layers_complete += 1`**
    - Increments counter to track progress
  - **If layers_complete == total_layers:**
    - **All layers done for this iteration**
    - **[connector_worker.py:140](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L140) `event_sync_blocking()`**
      - Waits for all CUDA events to complete
      - Ensures GPU writes are visible to CPU
      - **[connector/worker.rs:245](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L245) `wait_for_gpu_completion()`**
        - Synchronizes each event
        - Guarantees data consistency
    - **[connector_worker.py:145](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L145) `enqueue_offloading_operations()`**
      - **For each operation in offloading_operations:**
        - **[connector/worker.rs:260](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L260) `enqueue_store_operation(operation)`**
          - **[connector/scheduler.rs:210](lib/llm/src/block_manager/connector/scheduler.rs#L210) `WorkerSchedulerClient::submit_worker_request()`**
            - Creates `WorkerTransferRequest` with:
              - `transfer_type`: Store
              - Other fields same as Load
            - Sends to Scheduler
            - Scheduler matches with `LeaderTransferRequest`
          - Adds request_id to `self.maybe_finished_offloading`
      - **[connector_worker.py:150](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L150) `offloading_operations.clear()`**
        - Releases buffer for next iteration

### 4. Completion Polling

- **[dynamo_connector.py:133](lib/kvbm/python/kvbm/vllm_integration/connector/dynamo_connector.py#L133) `get_finished(finished_req_ids)`**
  - vLLM calls to check which transfers completed
  - `finished_req_ids`: Set of requests done generating
  - **For each request_id in maybe_finished_onboarding:**
    - **[connector_worker.py:165](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L165) `connector.is_complete(request_id)`**
      - **[connector/worker.rs:290](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L290) `is_complete(request_id)`**
        - **[connector/worker.rs:295](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L295) `check_completion_counter(request_id)`**
          - Reads `slot.completion_counter.load(Ordering::Acquire)`
          - Returns `true` if counter == num_operations
          - Indicates all Load operations finished
    - **If complete:**
      - Add to `is_finished_onboarding` result set
      - Remove from `maybe_finished_onboarding` tracking
  - **For each request_id in maybe_finished_offloading:**
    - Same completion check as onboarding
    - **If complete:**
      - Add to `is_finished_offloading` result set
      - **[connector_worker.py:175](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L175) `connector.remove_slot(request_id)`**
        - **[connector/worker.rs:315](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L315) `remove_slot(request_id)`**
          - Removes slot from tracking HashMap
          - Frees completion counter memory
  - Returns `(is_finished_onboarding, is_finished_offloading)` to vLLM
  - vLLM can now safely run forward pass for onboarded requests

---

## Transfer Coordination (Scheduler)

### Leader-Side Transfer Engine

- **[connector/leader.rs:335](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L335) `LocalTransferEngine::run()`**
  - Async event loop processing transfer requests
  - **When LocalOnboardRequest received:**
    - Request from slot's `trigger_onboarding()`
    - **[connector/leader.rs:355](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L355) `handle_onboard_request(request)`**
      - **[connector/leader.rs:365](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L365) `extract_source_blocks(src_blocks)`**
        - Looks up blocks in host/disk pools
        - Retrieves block metadata (location, size)
      - **[connector/leader.rs:375](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L375) `create_leader_transfer_request()`**
        - Creates `LeaderTransferRequest`:
          - `uuid`: Unique identifier
          - `src_blocks`: Host/disk locations
          - `dst_worker_id`: Target worker ID
          - `request_type`: Immediate (high priority)
        - **[connector/leader.rs:385](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L385) `leader.transfer_blocks_request(request)`**
          - **[connector_leader.py:180](lib/kvbm/python/kvbm/vllm_integration/connector_leader.py#L180) `KvbmLeader.transfer_blocks_request()`**
            - **[connector/scheduler.rs:450](lib/llm/src/block_manager/connector/scheduler.rs#L450) `TransferSchedulerClient::notify(leader_request)`**
              - Sends `LeaderTransferRequest` to Scheduler
              - Scheduler waits for matching worker request
  - **When LocalOffloadRequest received:**
    - Request from slot's `offload_blocks()`
    - **[connector/leader.rs:405](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L405) `handle_offload_request(request)`**
      - **[connector/leader.rs:415](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L415) `allocate_host_disk_blocks(token_blocks)`**
        - Allocates space in host/disk pools
        - Registers token metadata in cache
      - **[connector/leader.rs:425](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L425) `apply_token_blocks(token_blocks)`**
        - Updates cache index with semantic content
        - Enables future prefix matching
      - **[connector/leader.rs:435](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L435) `create_leader_transfer_request()`**
        - Creates `LeaderTransferRequest`:
          - `request_type`: Scheduled (normal priority)
        - Sends to Scheduler via `TransferSchedulerClient`

### Worker-Side Scheduler

- **[connector/scheduler.rs:100](lib/llm/src/block_manager/connector/scheduler.rs#L100) `Scheduler::run()`**
  - Async event loop coordinating transfers
  - **Receives WorkerTransferRequest from WorkerSchedulerClient:**
    - Sent by worker when metadata bound
    - Contains worker-side transfer details
    - Stored in `pending_worker_requests` HashMap by UUID
  - **Receives LeaderTransferRequest from TransferSchedulerClient:**
    - Sent by leader's transfer engine
    - Contains leader-side transfer details
    - Stored in `pending_leader_requests` HashMap by UUID
  - **When both requests present for same UUID:**
    - **[connector/scheduler.rs:140](lib/llm/src/block_manager/connector/scheduler.rs#L140) `match_and_schedule(uuid)`**
      - Removes both requests from pending maps
      - **[connector/scheduler.rs:155](lib/llm/src/block_manager/connector/scheduler.rs#L155) `validate_transfer_consistency()`**
        - Verifies src/dst blocks match
        - Checks transfer type compatibility
        - Ensures request_id consistency
      - **If valid:**
        - **[connector/scheduler.rs:180](lib/llm/src/block_manager/connector/scheduler.rs#L180) `create_task_controller()`**
          - Creates `ScheduledTaskController` for two-phase commit
          - **Phase 1: Prepare**
            - Allocates transfer resources
            - Sets up memory buffers
            - Validates block accessibility
          - **Phase 2: Execute**
            - **[connector/scheduler.rs:220](lib/llm/src/block_manager/connector/scheduler.rs#L220) `execute_transfer()`**
              - **If TransferType::Load:**
                - **[connector/scheduler.rs:230](lib/llm/src/block_manager/connector/scheduler.rs#L230) `transfer_client.load_blocks()`**
                  - Reads from host/disk blocks
                  - Writes to device blocks (GPU)
                  - Uses CUDA async memcpy
              - **If TransferType::Store:**
                - **[connector/scheduler.rs:250](lib/llm/src/block_manager/connector/scheduler.rs#L250) `transfer_client.store_blocks()`**
                  - Reads from device blocks (GPU)
                  - Writes to host/disk blocks
                  - Uses CUDA async memcpy
            - **[connector/scheduler.rs:270](lib/llm/src/block_manager/connector/scheduler.rs#L270) `signal_completion()`**
              - Increments worker's completion counter:
                - `completion_handle.fetch_add(1, Ordering::Release)`
              - Worker observes via `is_complete()` polling
              - Enables lock-free coordination
          - **If validation fails:**
            - **[connector/scheduler.rs:290](lib/llm/src/block_manager/connector/scheduler.rs#L290) `cancel_transfer()`**
              - Logs error details
              - Signals worker via completion counter (error value)
              - Cleans up resources

---

## Key Data Structures

### ConnectorMetadata (Serialized)

**Defined in [connector/mod.rs](lib/kvbm/src/block_manager/vllm/connector/mod.rs)**

```rust
struct ConnectorMetadata {
    new_slots: Vec<String>,              // Request IDs to create
    transfer_operations: Vec<Operation>, // Load/Store operations
    slot_state_updates: Vec<StateUpdate> // State synchronization
}

struct Operation {
    request_id: String,
    transfer_type: TransferType, // Load or Store
    src_blocks: Vec<BlockId>,
    dst_blocks: Vec<BlockId>,
    uuid: Uuid,                  // For scheduler matching
    token_blocks: Option<Vec<TokenBlock>> // For Store operations
}
```

### VllmConnectorSlot State Machine

**Defined in [connector/vllm_connector_slot.rs](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs)**

```
Initialized
    |
    v
OnboardStaged(n) --> trigger_onboarding()
    |
    v
Onboarding(n) --> is_complete()
    |
    v
Prefilling --> apply_scheduler_output()
    |                  |
    v                  v
Decoding         SkippedPrefill
    |                  |
    v                  v
SkippedDecode    (offload triggered)
    |                  |
    v                  v
Finishing --> is_complete()
    |
    v
Finished
```

### Transfer Request Types

**Defined in [connector/scheduler.rs](lib/llm/src/block_manager/connector/scheduler.rs)**

```rust
// Worker-side request
struct WorkerTransferRequest {
    uuid: Uuid,                    // Matches LeaderTransferRequest
    request_id: String,
    src_blocks: Vec<BlockId>,
    dst_blocks: Vec<BlockId>,
    transfer_type: TransferType,
    completion_handle: Arc<AtomicU64> // For lock-free completion
}

// Leader-side request
struct LeaderTransferRequest {
    uuid: Uuid,                    // Matches WorkerTransferRequest
    src_blocks: Vec<BlockLocation>,
    dst_worker_id: u32,
    request_type: RequestType,     // Immediate vs Scheduled
    token_blocks: Option<Vec<TokenBlock>>
}
```

---

## Synchronization Mechanisms

### 1. Metadata Broadcast Sync

- **Leader serializes ConnectorMetadata → vLLM broadcasts via RPC → Workers deserialize**
- Ensures all workers have consistent view of operations
- Happens once per iteration before forward pass

### 2. Two-Phase Transfer Coordination

- **Worker submits WorkerTransferRequest → Scheduler waits → Leader submits LeaderTransferRequest**
- Scheduler matches by UUID and executes
- Prevents race conditions in distributed transfers

### 3. Lock-Free Completion Tracking

- **Scheduler increments AtomicU64 → Worker polls via load(Acquire)**
- Avoids mutex overhead for high-frequency checks
- Memory ordering ensures visibility

### 4. GPU Event Synchronization

- **Worker records CUDA events → Syncs before offload → Ensures data consistency**
- Prevents offloading stale GPU data
- Critical for correctness of cached blocks

---

## Performance Optimizations

### 1. Immediate vs Scheduled Transfers

- **Onboarding (Load)**: `RequestType::Immediate` - high priority, blocks forward pass
- **Offloading (Store)**: `RequestType::Scheduled` - normal priority, async
- Prioritizes latency-critical operations

### 2. Batched GPU Synchronization

- Worker waits for all layers to complete before syncing
- Single sync point instead of per-layer
- Reduces CPU-GPU synchronization overhead

### 3. Lazy Worker Initialization

- Worker components created on first `register_kv_caches()` call
- Avoids initialization for rank-0 leader-only processes
- Reduces memory footprint

### 4. Async Task Architecture

- LocalTransferEngine and Scheduler run as Tokio tasks
- Non-blocking coordination channels (mpsc)
- Overlaps transfer preparation with GPU execution

---

## Error Handling

### Transfer Validation Errors

- **[connector/scheduler.rs:155](lib/llm/src/block_manager/connector/scheduler.rs#L155) `validate_transfer_consistency()`**
  - Catches block ID mismatches
  - Detects incompatible transfer types
  - Logs error and cancels operation
  - Signals worker via completion counter

### GPU Synchronization Errors

- **[connector/worker.rs:245](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L245) `wait_for_gpu_completion()`**
  - Catches CUDA event errors
  - Aborts offload if GPU operation failed
  - Prevents corrupted data in cache

### ZMQ Communication Errors

- **[connector_worker.py:88](lib/kvbm/python/kvbm/vllm_integration/connector_worker.py#L88) `KvbmWorker.new()`**
  - Connection timeout handling
  - Retries with exponential backoff
  - Fails gracefully if leader unreachable

### Slot State Errors

- **[connector/vllm_connector_slot.rs:330](lib/kvbm/src/block_manager/vllm/connector/vllm_connector_slot.rs#L330) `apply_scheduler_output()`**
  - Validates state transitions
  - Catches invalid state machine progressions
  - Logs warning and recovers to safe state

---

## Metrics and Observability

### Leader Metrics

- **[connector/leader.rs:138](lib/kvbm/src/block_manager/vllm/connector/leader.rs#L138) `metrics` parameter**
  - Onboarding operation count and latency
  - Offloading operation count and latency
  - Cache hit rate per request
  - Block allocation/deallocation rates

### Worker Metrics

- **[connector/worker.rs:89](lib/kvbm/src/block_manager/vllm/connector/worker.rs#L89) `worker_metrics` in config**
  - Transfer operation count by type (Load/Store)
  - GPU synchronization time
  - Scheduler queue depths
  - Completion polling frequency

### Transfer Metrics

- **[connector/scheduler.rs:220](lib/llm/src/block_manager/connector/scheduler.rs#L220) `execute_transfer()` instrumentation**
  - Bytes transferred per operation
  - Transfer bandwidth (device ↔ host ↔ disk)
  - Scheduler matching latency
  - Two-phase commit duration
