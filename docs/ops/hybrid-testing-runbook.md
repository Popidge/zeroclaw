# Hybrid Testing Runbook

Purpose: validate the edge router + edge executor + frontier executor flow on a real machine.

Audience: code agents and operators running ZeroClaw locally.

Last reviewed: **March 12, 2026**.

## Scope

Use this runbook when testing:

- `provider = "hybrid"`
- local router behavior
- local edge execution behavior
- frontier fallback behavior
- follow-up turn routing behavior

This document is for interactive validation, not benchmarking methodology or training work.

## Assumptions

- ZeroClaw is built and installed on the target machine.
- A local OpenAI-compatible endpoint is running for the edge model.
  - Example: LM Studio developer server on `http://127.0.0.1:1234`
- A frontier provider is configured and reachable.
- The repo checkout contains the hybrid harness branch or a descendant of it.

## Quick Context

Current hybrid behavior is intentionally conservative:

- router runs only on inbound user turns
- router input is heavily compacted for latency
- large local prompt estimates bypass router and pin `frontier`
- frontier decisions can lease for a few follow-up turns
- host-action tools are treated as edge-only; frontier should delegate them back to edge

Current known weak points:

- small local models may still time out in the router role on weaker hardware
- edge executor quality is more fragile than frontier for exact-output tasks
- if `~/.zeroclaw/config.toml` still contains `hybrid.classifier.temperature`, remove it because it is stale

## Baseline Config

Use a config shaped like this:

```toml
default_provider = "hybrid"
default_model = "hybrid:auto"

[hybrid]
enabled = true

[hybrid.edge]
provider = "lmstudio"
model = "qwen3.5-0.8b"

[hybrid.frontier]
provider = "anthropic-custom:https://api.kimi.com/coding/"
model = "kimi-for-coding"

[router]
# optional overrides; by default router inherits edge provider/model
# model = "qwen3.5-0.8b"
# timeout_ms = 4000
```

If testing a stronger machine, a useful next setup is:

- router: `qwen3.5-0.8b`
- edge executor: `qwen3.5-2b` or `qwen3.5-4b`

## Baseline Checks

1. Verify the branch and binary:

```bash
git branch --show-current
zeroclaw --version
```

2. Verify local edge model availability:

```bash
curl -sS http://127.0.0.1:1234/v1/models
```

3. Verify raw local inference before blaming ZeroClaw:

```bash
curl -sS http://127.0.0.1:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"Reply with exactly EDGE_DIRECT_OK"}],"temperature":0}'
```

Expected result:

- local endpoint returns quickly
- response contains `EDGE_DIRECT_OK`

If this step is slow, fix the local model runtime first.

## Main Probe Sequence

Run these in order.

### 1. Forced Frontier

```bash
zeroclaw agent --model hybrid:frontier -m "Reply with exactly FRONTIER_FORCED_OK"
```

Expected:

- returns `FRONTIER_FORCED_OK`
- proves the frontier leg is reachable

### 2. Forced Edge

```bash
zeroclaw agent --model hybrid:edge -t 0 -m "Reply with exactly EDGE_FORCED_OK"
```

Expected:

- completes without hanging
- ideally returns `EDGE_FORCED_OK`

Interpretation:

- if it hangs, local execution is still too heavy or the provider path is broken
- if it returns but ignores exact output, local quality/prompt contract needs work

### 3. Auto Simple Turn

```bash
zeroclaw agent -m "Reply with exactly AUTO_SIMPLE_OK"
```

Expected best case:

- router chooses `edge`
- response is fast and exact

Acceptable current fallback:

- router times out or is low confidence
- request falls through to `frontier`
- result is still correct

### 4. Auto Complex Turn

```bash
zeroclaw agent -m "Give me a 3-step plan to migrate a Rust web app from Actix to Axum."
```

Expected:

- routes to `frontier`
- result is coherent and concise

Note:

- this turn may trigger `delegate` in supervised mode depending on prompt shape and local tool policy
- if that interferes with routing validation, use a complex but purely conversational prompt instead

Example conversational alternative:

```bash
zeroclaw agent -m "What are the main tradeoffs between monolith and microservices for a 30-person engineering team?"
```

### 5. Continuation Behavior

Use an interactive session:

```bash
zeroclaw agent
```

Then send:

```text
What are the main tradeoffs between monolith and microservices for a 30-person engineering team?
Now reply with exactly FOLLOWUP_OK
/quit
```

Expected:

- first turn should go `frontier`
- second turn should still succeed cleanly
- if lease behavior is active, follow-up may stay on `frontier` without consulting the router again

## Optional Timing Wrapper

On shells with builtin `time`:

```bash
bash -lc 'TIMEFORMAT="ELAPSED=%R"; time zeroclaw agent -m "Reply with exactly AUTO_SIMPLE_OK"'
```

Track at least:

- raw local inference time
- forced edge time
- forced frontier time
- auto simple time

Useful interpretation:

- if forced edge is faster than forced frontier, the local executor path is viable
- if auto simple is slower than frontier because of router timeout, router budget is still the bottleneck

## Success Criteria

Consider the run positive if most of the following are true:

- local raw inference is healthy
- forced edge no longer hangs
- forced frontier is correct
- auto mode behaves safely
- simple turns are at least approaching frontier latency
- complex turns prefer frontier
- follow-up turns remain coherent

On weaker laptops, even a small wall-clock win for forced edge is meaningful.

On stronger hardware, the main target is:

- `hybrid:auto` should beat plain frontier on simple edge-capable turns

## Common Failure Modes

### Symptom: `router_timeout`

Cause:

- router prompt is still too expensive for the local model or timeout is too low

Fix:

- verify raw local inference first
- raise `[router].timeout_ms` modestly
- use a smaller router model
- reduce local prompt pressure further

### Symptom: forced edge hangs or is extremely slow

Cause:

- local runtime is overloaded
- edge prompt is still too large
- model is too large for the hardware/runtime configuration

Fix:

- test raw local inference outside ZeroClaw
- switch to a smaller edge model
- verify GPU offload / runtime config in LM Studio or llama.cpp

### Symptom: forced edge replies with commentary instead of exact output

Cause:

- edge executor prompt/contract is too loose

Fix:

- tighten edge executor prompt
- reduce available tool surface
- prefer structured output or schema enforcement when supported

### Symptom: auto mode always falls through to frontier

Cause:

- router timeout
- low router confidence
- prompt-size bypass
- frontier lease from prior complex turn

Fix:

- inspect logs
- test with a fresh session
- remove continuation effects before retesting

## Logging Tips

Useful signals come from info logs such as:

- `Pinned hybrid turn route`
- `reason_code="router_timeout"`
- `reason_code="prompt_too_large"`
- explicit `edge` or `frontier` target selection

If observability is disabled in config, basic stdout/stderr logs are still enough for the manual probes above.

## Recommended Agent Workflow

When another code agent picks this up on a new machine:

1. Read this file first.
2. Verify local model health with raw `curl`.
3. Run forced frontier, then forced edge, then auto simple.
4. Run one conversational complex turn and one continuation test.
5. Record:
   - wall clock time
   - route/fallback reason
   - output quality
6. Only after that, start changing router timeouts, prompts, or model sizes.

This avoids tuning blind.

## Related Docs

- [README.md](./README.md)
- [operations-runbook.md](./operations-runbook.md)
- [troubleshooting.md](./troubleshooting.md)
- [config-reference.md](../reference/api/config-reference.md)
