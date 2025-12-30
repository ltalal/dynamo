# KVBM Stuck Requests Investigation

## Problem Description

Requests are getting stuck in the `kvbm_connector_maybe_finished_offloading` and `kvbm_connector_maybe_finished_onboarding` metrics. Even after stopping request traffic, these metrics remain non-zero:

```
kvbm_connector_maybe_finished_offloading 1810
kvbm_connector_maybe_finished_onboarding 1
```

Related warning observed in logs:
```
WARNING 12-29 20:04:56 [scheduler.py:1292] Got finished sending KV transfer for request cmpl-0364d9a7f95c41e6b7e80f882af97429-0, but the request is already freed.
```

## Root Cause

The issue stems from a **race condition between request cleanup and in-flight operations** in a distributed system.

### Architecture

```
┌─────────────┐ ZMQ       ┌──────────────────┐
│   Leader    │────────>  │  Worker Process  │
│  Process    │           │  ┌──────────┐    │
│             │           │  │Scheduler │    │ (local)
│             │           │  └──────────┘    │
└─────────────┘           └──────────────────┘
```

- **Leader** and **Worker** are separate processes communicating via ZeroMQ
- Each has its own local state for tracking requests and operations

### The Bug Sequence

1. **Request starts**: Operations are enqueued for onboarding/offloading
   - Worker-side: Request ID added to `maybe_finished_*` HashSet
   - Scheduler: Operations added to `slot.operations` list

2. **Request gets freed/aborted** (user cancellation, error, etc.)
   - Leader-side: Slot removed (`trtllm_leader.rs:401`)
   - **Worker-side: NOT notified** ⚠️

3. **In-flight operations complete**
   - Completions arrive but slot is already gone on leader
   - vLLM logs: "Got finished sending KV transfer for request X, but the request is already freed"
   - **Completions are dropped** - never reach worker scheduler

4. **Worker-side stays stuck**
   - Slot exists with `N` operations in `slot.operations`
   - Only `M < N` completions received (some were dropped)
   - `slot.is_complete()` returns `false` forever because `completed < operations.len()`
   - Request remains in `maybe_finished_*` set indefinitely

### Code Evidence

#### The TODO Comment

In `lib/kvbm/src/block_manager/vllm/connector/trtllm_leader.rs:395-398`:

```rust
// todo: allow the request to resolve when it should exit
// the request may have some outstanding operations
// we would like to inform it to shutdown, then have it signal to the work that is officially gone,
// then we can remove the slot and trigger the worker to clean up as well.

// remove it from the manager as we will never use it again
self.slot_manager().remove_slot(&request_id)?;
```

**This is a known TODO** - the leader removes slots without notifying the worker to clean up.

#### Cancelled Operations Don't Count as Complete

In `lib/llm/src/block_manager/connector/protocol.rs:290-300`:

```rust
pub struct CancelledTransferCompletionHandle;

impl TransferCompletionHandle for CancelledTransferCompletionHandle {
    fn scheduler_decision(&self) -> SchedulingDecision {
        SchedulingDecision::Cancel
    }

    async fn mark_complete(&self, _result: anyhow::Result<()>) {
        // Do nothing  ← BUG: Never increments completed counter!
    }
}
```

When operations are cancelled, they don't increment the `completed` counter, but they're still counted in `operations.len()`.

#### Completion Check Logic

In `lib/llm/src/block_manager/connector/scheduler.rs:223-230`:

```rust
pub fn is_complete(&self, request_id: &str) -> bool {
    match self.slots.get(request_id) {
        Some(slot) => slot.completed.load(Ordering::Relaxed) == slot.operations.len() as u64,
        None => {
            tracing::debug!(request_id, "slot not found - likely aborted");
            true
        }
    }
}
```

A slot is only complete when `completed == operations.len()`. If any operations:
- Are cancelled without incrementing `completed`
- Complete after the slot is removed (completions dropped)
- Fail silently

Then the slot will **never** be considered complete.

#### Worker Cleanup Logic

In `lib/kvbm/src/block_manager/vllm/connector/worker.rs:405-420`:

```rust
// visit each request slot in the maybe finished set
for request_id in self.maybe_finished_offloading.iter() {
    if self.connector.has_slot(request_id) {
        if self.connector.is_complete(request_id) {
            tracing::debug!(request_id, "request slot is finished");
            is_finished_offloading.insert(request_id.clone());
        } else {
            tracing::debug!(request_id, "request slot is not finished");
        }
    } else {
        panic!(
            "request slot missing for {request_id}; however, it was present when added to the maybe finished offloading set"
        );
    }
}
```

Requests are only removed from `maybe_finished_*` when `is_complete()` returns true. If it never returns true, they stay forever.

## Proposed Solutions

### Option 1: Add ZMQ Notification (Proper Fix)

**Pros:**
- Addresses root cause
- Clean architecture
- Immediate cleanup

**Cons:**
- Requires protocol change
- Needs coordination across distributed processes
- More invasive change

**Implementation:**
1. Add new message type for request cleanup notification
2. Leader sends message when removing slot
3. Worker scheduler receives message and force-completes or removes slot
4. Worker removes request from `maybe_finished_*` sets

### Option 2: Timeout-Based Cleanup (Simple Workaround)

**Pros:**
- No protocol changes needed
- Simple to implement
- Local to worker code
- Can be deployed immediately

**Cons:**
- Doesn't address root cause
- Arbitrary timeout value
- Delayed cleanup

**Implementation:**

In `lib/kvbm/src/block_manager/vllm/connector/worker.rs`, track how long requests have been stuck:

```rust
// Add field to KvConnectorWorker
stuck_request_iterations: HashMap<String, u64>,

// In get_finished() method:
const STUCK_THRESHOLD: u64 = 100; // ~100 forward passes
let mut force_cleanup = HashSet::new();

for request_id in self.maybe_finished_offloading.iter() {
    if self.connector.has_slot(request_id) {
        if self.connector.is_complete(request_id) {
            is_finished_offloading.insert(request_id.clone());
            self.stuck_request_iterations.remove(request_id);
        } else {
            // Track stuck iterations
            let stuck_count = self.stuck_request_iterations
                .entry(request_id.clone())
                .or_insert(0);
            *stuck_count += 1;

            if *stuck_count > STUCK_THRESHOLD {
                tracing::warn!(
                    request_id,
                    stuck_iterations = stuck_count,
                    "Force cleaning up stuck request that has been incomplete for too long"
                );
                force_cleanup.insert(request_id.clone());
                self.stuck_request_iterations.remove(request_id);
            }
        }
    }
}

// Force cleanup stuck requests
for request_id in &force_cleanup {
    self.maybe_finished_offloading.remove(request_id);
    if self.connector.has_slot(request_id) {
        self.connector.remove_slot(request_id);
    }
}
```

### Option 3: Fix Cancelled Operation Accounting

Make cancelled operations count as "complete":

In `lib/llm/src/block_manager/connector/protocol.rs`:

```rust
impl TransferCompletionHandle for CancelledTransferCompletionHandle {
    async fn mark_complete(&self, _result: anyhow::Result<()>) {
        // Still increment the completed counter for cancelled operations
        // They're "done" even if cancelled
    }
}
```

**Note:** This only fixes cancelled operations, not the race condition with freed requests.

## Recommendation

**Start with Option 2** (timeout-based cleanup) as an immediate workaround:
- Simple to implement and test
- No distributed coordination needed
- Prevents unbounded memory growth
- Can be deployed quickly

**Then implement Option 1** (proper ZMQ notification):
- Proper long-term fix
- Addresses root cause
- Implement the TODO from line 395-398

**Also apply Option 3**:
- Small fix with no downsides
- Cancelled operations should count as complete

## Files Involved

- `lib/kvbm/src/block_manager/vllm/connector/worker.rs` - Worker-side logic, `maybe_finished_*` sets
- `lib/kvbm/src/block_manager/vllm/connector/trtllm_leader.rs` - Leader-side cleanup with TODO
- `lib/llm/src/block_manager/connector/scheduler.rs` - Scheduler completion tracking
- `lib/llm/src/block_manager/connector/protocol.rs` - Cancelled operation handling
