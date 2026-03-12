use crate::config::{HybridProviderConfig, HybridRouteTarget, ReliabilityConfig, RouterConfig};
use crate::multimodal;
use crate::observability::runtime_trace;
use crate::providers::{self, ChatMessage, ProviderRuntimeOptions};
use crate::util::truncate_with_ellipsis;
use anyhow::Context;
use serde::Deserialize;

const ROUTER_HARD_MAX_CHARS: usize = 1_200;
const ROUTER_HARD_MAX_MESSAGES: usize = 4;
const ROUTER_MESSAGE_CHAR_BUDGET: usize = 280;
const EDGE_PROMPT_BYPASS_CHARS: usize = 2_400;
pub(crate) const FRONTIER_ROUTE_LEASE_TURNS: usize = 2;
const FRONTIER_EDGE_ONLY_TOOLS: &[&str] = &[
    "shell",
    "file_read",
    "file_write",
    "file_edit",
    "glob_search",
    "content_search",
    "pdf_read",
    "git_operations",
    "browser",
    "browser_open",
    "screenshot",
    "image_info",
    "proxy_config",
];

#[derive(Debug, Clone, PartialEq)]
pub struct TurnRouteDecision {
    pub target: HybridRouteTarget,
    pub provider_name: String,
    pub model: String,
    pub confidence: f64,
    pub reason_code: String,
    pub summary: String,
    pub delegated_from_router: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct RouterDecisionPayload {
    target: String,
    confidence: f64,
    reason_code: String,
    #[serde(default)]
    summary: String,
}

#[derive(Debug, Clone, Copy)]
pub struct RouterConversationContext {
    pub had_prior_history: bool,
    pub frontier_lease_remaining: usize,
}

#[derive(Debug, Clone)]
pub struct HybridRouterRequest<'a> {
    pub requested_provider: &'a str,
    pub requested_model: &'a str,
    pub global_api_key: Option<&'a str>,
    pub reliability: &'a ReliabilityConfig,
    pub hybrid: &'a HybridProviderConfig,
    pub router: &'a RouterConfig,
    pub provider_runtime_options: &'a ProviderRuntimeOptions,
    pub history: &'a [ChatMessage],
    pub channel_name: &'a str,
    pub conversation: RouterConversationContext,
}

pub fn hybrid_requires_prompt_tool_instructions(provider_name: &str) -> bool {
    provider_name.trim().eq_ignore_ascii_case("hybrid")
}

pub fn frontier_edge_only_excluded_tools(existing: &[String]) -> Vec<String> {
    let mut merged = existing.to_vec();
    for tool in FRONTIER_EDGE_ONLY_TOOLS {
        if !merged.iter().any(|value| value == tool) {
            merged.push((*tool).to_string());
        }
    }
    merged
}

pub fn next_frontier_lease_after_turn(
    decision: &TurnRouteDecision,
    previous_remaining: usize,
) -> usize {
    if decision.reason_code == "frontier_lease" {
        previous_remaining.saturating_sub(1)
    } else if matches!(decision.target, HybridRouteTarget::Frontier) {
        FRONTIER_ROUTE_LEASE_TURNS
    } else {
        0
    }
}

pub async fn resolve_turn_route(
    request: HybridRouterRequest<'_>,
) -> anyhow::Result<Option<TurnRouteDecision>> {
    if !request
        .requested_provider
        .trim()
        .eq_ignore_ascii_case("hybrid")
    {
        return Ok(None);
    }

    if !request.hybrid.enabled {
        anyhow::bail!("provider 'hybrid' requires [hybrid].enabled = true");
    }

    if let Some(explicit_target) =
        crate::providers::hybrid::HybridProvider::parse_explicit_target(request.requested_model)
    {
        return Ok(Some(resolve_explicit_decision(
            request.hybrid,
            explicit_target,
            request.global_api_key,
            "explicit_model",
            "Pinned by explicit hybrid model selection",
        )?));
    }

    if multimodal::count_image_markers(request.history) > 0 {
        return Ok(Some(resolve_explicit_decision(
            request.hybrid,
            HybridRouteTarget::Frontier,
            request.global_api_key,
            "vision_required",
            "Frontier selected because the turn contains image input",
        )?));
    }

    if request.conversation.frontier_lease_remaining > 0 {
        return Ok(Some(resolve_explicit_decision(
            request.hybrid,
            HybridRouteTarget::Frontier,
            request.global_api_key,
            "frontier_lease",
            &format!(
                "Conversation remains pinned to frontier for {} more follow-up turn(s)",
                request.conversation.frontier_lease_remaining
            ),
        )?));
    }

    let estimated_prompt_chars = estimate_router_working_set_chars(request.history);
    if estimated_prompt_chars > EDGE_PROMPT_BYPASS_CHARS {
        return Ok(Some(resolve_explicit_decision(
            request.hybrid,
            HybridRouteTarget::Frontier,
            request.global_api_key,
            "prompt_too_large",
            &format!(
                "Bypassed router because local prompt estimate ({estimated_prompt_chars} chars) exceeds edge budget ({EDGE_PROMPT_BYPASS_CHARS})"
            ),
        )?));
    }

    let router_target = providers::resolve_router_target_from_config(
        request.router,
        request.hybrid,
        request.global_api_key,
    )?;
    let router_provider = providers::create_router_provider_from_resolved(
        &router_target,
        request.reliability,
        request.provider_runtime_options,
    )?;

    let router_prompt = build_router_prompt(request.history, request.router, request.conversation);
    let router_system_prompt = request
        .router
        .system_prompt
        .as_deref()
        .unwrap_or(crate::providers::hybrid::ROUTER_DEFAULT_SYSTEM_PROMPT);

    let router_result = tokio::time::timeout(
        std::time::Duration::from_millis(request.router.timeout_ms),
        router_provider.chat_with_system(
            Some(router_system_prompt),
            &router_prompt,
            &router_target.model,
            request.router.temperature,
        ),
    )
    .await;

    let (decision, delegated_from_router) = match router_result {
        Ok(Ok(raw)) => {
            let parsed = parse_router_response(&raw).with_context(|| {
                format!(
                    "router returned invalid output: {}",
                    truncate_with_ellipsis(&raw, 240)
                )
            });
            match parsed {
                Ok(parsed) if parsed.confidence >= request.router.confidence_threshold => {
                    (parsed, true)
                }
                Ok(parsed) => (
                    RouterDecisionPayload {
                        target: "frontier".to_string(),
                        confidence: parsed.confidence,
                        reason_code: "low_confidence".to_string(),
                        summary: format!(
                            "Router confidence {:.2} is below threshold {:.2}",
                            parsed.confidence, request.router.confidence_threshold
                        ),
                    },
                    false,
                ),
                Err(error) => (
                    RouterDecisionPayload {
                        target: "frontier".to_string(),
                        confidence: 0.0,
                        reason_code: "router_invalid".to_string(),
                        summary: error.to_string(),
                    },
                    false,
                ),
            }
        }
        Ok(Err(error)) => (
            RouterDecisionPayload {
                target: "frontier".to_string(),
                confidence: 0.0,
                reason_code: "router_error".to_string(),
                summary: providers::sanitize_api_error(&error.to_string()),
            },
            false,
        ),
        Err(_) => (
            RouterDecisionPayload {
                target: "frontier".to_string(),
                confidence: 0.0,
                reason_code: "router_timeout".to_string(),
                summary: format!("Router timed out after {}ms", request.router.timeout_ms),
            },
            false,
        ),
    };

    let target = parse_target(&decision.target).unwrap_or(HybridRouteTarget::Frontier);
    let resolved = providers::resolve_hybrid_target_from_config(
        request.hybrid,
        target,
        request.global_api_key,
    )?;

    tracing::info!(
        route = match target {
            HybridRouteTarget::Edge => "edge",
            HybridRouteTarget::Frontier => "frontier",
        },
        confidence = decision.confidence,
        reason_code = decision.reason_code.as_str(),
        delegated_from_router,
        channel = request.channel_name,
        "Pinned hybrid turn route"
    );
    runtime_trace::record_event(
        "router_decision",
        Some(request.channel_name),
        Some(router_target.provider_name.as_str()),
        Some(router_target.model.as_str()),
        None,
        Some(matches!(target, HybridRouteTarget::Edge)),
        None,
        serde_json::json!({
            "target": match target {
                HybridRouteTarget::Edge => "edge",
                HybridRouteTarget::Frontier => "frontier",
            },
            "confidence": decision.confidence,
            "reason_code": decision.reason_code,
            "summary": decision.summary,
            "had_prior_history": request.conversation.had_prior_history,
            "frontier_lease_remaining": request.conversation.frontier_lease_remaining,
            "recent_messages": request.history.len(),
            "estimated_prompt_chars": estimated_prompt_chars,
            "delegated_from_router": delegated_from_router,
            "execution_provider": resolved.provider_name,
            "execution_model": resolved.model,
        }),
    );

    Ok(Some(TurnRouteDecision {
        target,
        provider_name: resolved.provider_name,
        model: resolved.model,
        confidence: decision.confidence,
        reason_code: decision.reason_code,
        summary: decision.summary,
        delegated_from_router,
    }))
}

fn resolve_explicit_decision(
    hybrid: &HybridProviderConfig,
    target: HybridRouteTarget,
    global_api_key: Option<&str>,
    reason_code: &str,
    summary: &str,
) -> anyhow::Result<TurnRouteDecision> {
    let resolved = providers::resolve_hybrid_target_from_config(hybrid, target, global_api_key)?;
    Ok(TurnRouteDecision {
        target,
        provider_name: resolved.provider_name,
        model: resolved.model,
        confidence: 1.0,
        reason_code: reason_code.to_string(),
        summary: summary.to_string(),
        delegated_from_router: false,
    })
}

fn parse_router_response(raw: &str) -> anyhow::Result<RouterDecisionPayload> {
    if let Ok(parsed) = serde_json::from_str::<RouterDecisionPayload>(raw) {
        return Ok(parsed);
    }

    if let (Some(start), Some(end)) = (raw.find('{'), raw.rfind('}')) {
        if start < end {
            return serde_json::from_str::<RouterDecisionPayload>(&raw[start..=end])
                .context("failed to parse embedded router JSON");
        }
    }

    anyhow::bail!("router did not return valid JSON")
}

fn parse_target(raw: &str) -> Option<HybridRouteTarget> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "edge" => Some(HybridRouteTarget::Edge),
        "frontier" => Some(HybridRouteTarget::Frontier),
        _ => None,
    }
}

fn build_router_prompt(
    history: &[ChatMessage],
    config: &RouterConfig,
    conversation: RouterConversationContext,
) -> String {
    let compact_messages: Vec<String> = history
        .iter()
        .filter(|message| message.role != "system")
        .map(|message| {
            format!(
                "{}: {}",
                message.role.to_uppercase(),
                compact_message_for_local_model(&message.content, ROUTER_MESSAGE_CHAR_BUDGET)
            )
        })
        .filter(|line| !line.ends_with(": "))
        .collect();
    let selected_messages =
        if compact_messages.len() > config.max_messages.min(ROUTER_HARD_MAX_MESSAGES) {
            &compact_messages
                [compact_messages.len() - config.max_messages.min(ROUTER_HARD_MAX_MESSAGES)..]
        } else {
            compact_messages.as_slice()
        };

    let recent_non_system = selected_messages.len();
    let estimated_recent_turns = recent_non_system.div_ceil(2);
    let transcript = truncate_with_ellipsis(
        &selected_messages.join("\n"),
        config.max_chars.min(ROUTER_HARD_MAX_CHARS),
    );

    format!(
        "Route this user turn.\n\nMetadata:\n- has_prior_history: {}\n- conversation_stage: {}\n- estimated_recent_turns: {}\n- frontier_lease_remaining: {}\n\nRecent transcript:\n{}\n",
        conversation.had_prior_history,
        if conversation.had_prior_history {
            "continuation"
        } else {
            "new_chat"
        },
        estimated_recent_turns,
        conversation.frontier_lease_remaining,
        transcript,
    )
}

pub(crate) fn compact_message_for_local_model(content: &str, max_chars: usize) -> String {
    let mut compacted_lines = Vec::new();
    let mut skipping_memory_block = false;

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            if skipping_memory_block {
                skipping_memory_block = false;
            }
            continue;
        }

        if line.eq_ignore_ascii_case("[memory context]") {
            skipping_memory_block = true;
            continue;
        }
        if skipping_memory_block && line.starts_with("- ") {
            continue;
        }
        if line.starts_with("[Used tools:") {
            continue;
        }

        let normalized = if line.starts_with('[') && line.contains("] ") {
            line.split_once("] ")
                .map(|(_, tail)| tail)
                .unwrap_or(line)
                .trim()
        } else {
            line
        };
        if !normalized.is_empty() {
            compacted_lines.push(normalized);
        }
    }

    truncate_with_ellipsis(&compacted_lines.join(" "), max_chars)
}

fn estimate_router_working_set_chars(history: &[ChatMessage]) -> usize {
    history
        .iter()
        .filter(|message| message.role != "system")
        .map(|message| {
            compact_message_for_local_model(&message.content, usize::MAX)
                .chars()
                .count()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_frontier_excluded_tools_without_duplicates() {
        let merged = frontier_edge_only_excluded_tools(&["delegate".into(), "shell".into()]);
        assert!(merged.iter().any(|tool| tool == "delegate"));
        assert_eq!(
            merged
                .iter()
                .filter(|tool| tool.as_str() == "shell")
                .count(),
            1
        );
        assert!(merged.iter().any(|tool| tool == "file_read"));
    }

    #[test]
    fn parses_embedded_router_json() {
        let parsed = parse_router_response(
            "result: {\"target\":\"edge\",\"confidence\":0.91,\"reason_code\":\"short_transform\",\"summary\":\"small\"}",
        )
        .expect("router json should parse");
        assert_eq!(parsed.target, "edge");
        assert_eq!(parsed.reason_code, "short_transform");
    }
}
