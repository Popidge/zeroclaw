use super::traits::{ChatMessage, ChatRequest, ChatResponse, ProviderCapabilities};
use super::Provider;
use crate::config::{HybridProviderConfig, HybridRouteTarget, HybridTargetConfig, RouterConfig};
use async_trait::async_trait;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectedTarget {
    Edge,
    Frontier,
}

impl From<HybridRouteTarget> for SelectedTarget {
    fn from(value: HybridRouteTarget) -> Self {
        match value {
            HybridRouteTarget::Edge => Self::Edge,
            HybridRouteTarget::Frontier => Self::Frontier,
        }
    }
}

impl From<SelectedTarget> for HybridRouteTarget {
    fn from(value: SelectedTarget) -> Self {
        match value {
            SelectedTarget::Edge => Self::Edge,
            SelectedTarget::Frontier => Self::Frontier,
        }
    }
}

pub const ROUTER_DEFAULT_SYSTEM_PROMPT: &str = "You are the routing controller for a hybrid AI assistant. Your only job is to classify the current user turn and return JSON only. Decide whether this turn should stay on the local edge executor or be handled by the frontier executor. Prefer edge only for brief, bounded requests with low ambiguity, low reasoning depth, and at most one straightforward local action. Prefer frontier for multi-step reasoning, planning, coding, research, comparisons, ambiguity, safety-sensitive judgement, or when confidence is not high. Return exactly this JSON schema: {\"target\":\"edge\"|\"frontier\",\"confidence\":0.0,\"reason_code\":\"trivial_reply|short_transform|host_action|multi_step|code_reasoning|ambiguous|safety_sensitive\",\"summary\":\"short explanation\"}.";

struct TargetProvider {
    provider_name: String,
    model: String,
    provider: Box<dyn Provider>,
}

pub struct HybridProvider {
    edge: TargetProvider,
    frontier: TargetProvider,
    fallback_target: SelectedTarget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridResolvedTarget {
    pub target: HybridRouteTarget,
    pub provider_name: String,
    pub model: String,
    pub api_key: Option<String>,
    pub api_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridResolvedRouter {
    pub provider_name: String,
    pub model: String,
    pub api_key: Option<String>,
    pub api_url: Option<String>,
}

impl HybridProvider {
    pub fn new(
        edge_provider_name: String,
        edge_model: String,
        edge_provider: Box<dyn Provider>,
        frontier_provider_name: String,
        frontier_model: String,
        frontier_provider: Box<dyn Provider>,
        fallback_target: HybridRouteTarget,
    ) -> Self {
        Self {
            edge: TargetProvider {
                provider_name: edge_provider_name,
                model: edge_model,
                provider: edge_provider,
            },
            frontier: TargetProvider {
                provider_name: frontier_provider_name,
                model: frontier_model,
                provider: frontier_provider,
            },
            fallback_target: fallback_target.into(),
        }
    }

    pub fn parse_explicit_target(model: &str) -> Option<HybridRouteTarget> {
        match model.trim().to_ascii_lowercase().as_str() {
            "hybrid:edge" => Some(HybridRouteTarget::Edge),
            "hybrid:frontier" => Some(HybridRouteTarget::Frontier),
            "hybrid:auto" | "" => None,
            _ => None,
        }
    }

    fn selected_provider(&self, target: SelectedTarget) -> (&str, &str, &dyn Provider) {
        match target {
            SelectedTarget::Edge => (
                self.edge.provider_name.as_str(),
                self.edge.model.as_str(),
                self.edge.provider.as_ref(),
            ),
            SelectedTarget::Frontier => (
                self.frontier.provider_name.as_str(),
                self.frontier.model.as_str(),
                self.frontier.provider.as_ref(),
            ),
        }
    }

    fn select_target(&self, requested_model: &str) -> SelectedTarget {
        Self::parse_explicit_target(requested_model)
            .map(SelectedTarget::from)
            .unwrap_or(self.fallback_target)
    }
}

pub fn resolve_hybrid_target(
    hybrid: &HybridProviderConfig,
    target: HybridRouteTarget,
    global_api_key: Option<&str>,
) -> anyhow::Result<HybridResolvedTarget> {
    if !hybrid.enabled {
        anyhow::bail!("provider 'hybrid' requires [hybrid].enabled = true");
    }

    let cfg = match target {
        HybridRouteTarget::Edge => &hybrid.edge,
        HybridRouteTarget::Frontier => &hybrid.frontier,
    };

    resolved_target_from_config(cfg, target, global_api_key)
}

pub fn resolve_router_target(
    router: &RouterConfig,
    hybrid: &HybridProviderConfig,
    global_api_key: Option<&str>,
) -> anyhow::Result<HybridResolvedRouter> {
    let edge = resolve_hybrid_target(hybrid, HybridRouteTarget::Edge, global_api_key)?;
    let provider_name = router
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(edge.provider_name.as_str())
        .to_string();
    let model = router
        .model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(edge.model.as_str())
        .to_string();

    if provider_name.trim().eq_ignore_ascii_case("hybrid") {
        anyhow::bail!("router.provider cannot be 'hybrid'");
    }
    if model.trim().is_empty() {
        anyhow::bail!(
            "router.model could not be resolved; set [router].model or [hybrid.edge].model"
        );
    }

    Ok(HybridResolvedRouter {
        provider_name,
        model,
        api_key: router
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .or(edge.api_key),
        api_url: router
            .api_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .or(edge.api_url),
    })
}

fn resolved_target_from_config(
    target: &HybridTargetConfig,
    route_target: HybridRouteTarget,
    global_api_key: Option<&str>,
) -> anyhow::Result<HybridResolvedTarget> {
    let provider_name = target.provider.trim();
    let model = target.model.trim();

    if provider_name.is_empty() {
        anyhow::bail!("hybrid target provider must not be empty");
    }
    if provider_name.eq_ignore_ascii_case("hybrid") {
        anyhow::bail!("hybrid target provider cannot itself be 'hybrid'");
    }
    if model.is_empty() {
        anyhow::bail!("hybrid target model must not be empty");
    }

    Ok(HybridResolvedTarget {
        target: route_target,
        provider_name: provider_name.to_string(),
        model: model.to_string(),
        api_key: target
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string)
            .or_else(|| global_api_key.map(ToString::to_string)),
        api_url: target
            .api_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToString::to_string),
    })
}

#[async_trait]
impl Provider for HybridProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: self.edge.provider.supports_native_tools()
                || self.frontier.provider.supports_native_tools(),
            vision: self.edge.provider.supports_vision()
                || self.frontier.provider.supports_vision(),
        }
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let target = self.select_target(model);
        let (provider_name, target_model, provider) = self.selected_provider(target);
        tracing::info!(
            provider = provider_name,
            model = target_model,
            route = match target {
                SelectedTarget::Edge => "edge",
                SelectedTarget::Frontier => "frontier",
            },
            "Hybrid provider dispatched one-shot request"
        );
        provider
            .chat_with_system(system_prompt, message, target_model, temperature)
            .await
    }

    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let target = self.select_target(model);
        let (provider_name, target_model, provider) = self.selected_provider(target);
        tracing::info!(
            provider = provider_name,
            model = target_model,
            route = match target {
                SelectedTarget::Edge => "edge",
                SelectedTarget::Frontier => "frontier",
            },
            "Hybrid provider dispatched history request"
        );
        provider
            .chat_with_history(messages, target_model, temperature)
            .await
    }

    async fn chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ChatResponse> {
        let target = self.select_target(model);
        let (provider_name, target_model, provider) = self.selected_provider(target);
        tracing::info!(
            provider = provider_name,
            model = target_model,
            route = match target {
                SelectedTarget::Edge => "edge",
                SelectedTarget::Frontier => "frontier",
            },
            "Hybrid provider dispatched structured chat request"
        );
        provider.chat(request, target_model, temperature).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ChatResponse> {
        let target = self.select_target(model);
        let (provider_name, target_model, provider) = self.selected_provider(target);
        tracing::info!(
            provider = provider_name,
            model = target_model,
            route = match target {
                SelectedTarget::Edge => "edge",
                SelectedTarget::Frontier => "frontier",
            },
            "Hybrid provider dispatched native-tools chat request"
        );
        provider
            .chat_with_tools(messages, tools, target_model, temperature)
            .await
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        if let Err(error) = self.edge.provider.warmup().await {
            tracing::warn!(
                provider = self.edge.provider_name.as_str(),
                "Hybrid edge warmup failed: {error}"
            );
        }
        if let Err(error) = self.frontier.provider.warmup().await {
            tracing::warn!(
                provider = self.frontier.provider_name.as_str(),
                "Hybrid frontier warmup failed: {error}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::ChatResponse;
    use parking_lot::Mutex;
    use std::sync::Arc;

    struct MockProvider {
        responses: Mutex<Vec<String>>,
        seen_models: Arc<Mutex<Vec<String>>>,
        supports_native_tools: bool,
        supports_vision: bool,
    }

    impl MockProvider {
        fn new(
            responses: Vec<&str>,
            seen_models: Arc<Mutex<Vec<String>>>,
            supports_native_tools: bool,
            supports_vision: bool,
        ) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().map(str::to_string).collect()),
                seen_models,
                supports_native_tools,
                supports_vision,
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                native_tool_calling: self.supports_native_tools,
                vision: self.supports_vision,
            }
        }

        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            self.seen_models.lock().push(model.to_string());
            Ok(self.responses.lock().remove(0))
        }

        async fn chat(
            &self,
            _request: ChatRequest<'_>,
            model: &str,
            _temperature: f64,
        ) -> anyhow::Result<ChatResponse> {
            self.seen_models.lock().push(model.to_string());
            Ok(ChatResponse {
                text: Some(self.responses.lock().remove(0)),
                tool_calls: Vec::new(),
                usage: None,
                reasoning_content: None,
            })
        }
    }

    #[tokio::test]
    async fn hybrid_auto_falls_back_to_configured_target() {
        let edge_seen = Arc::new(Mutex::new(Vec::new()));
        let frontier_seen = Arc::new(Mutex::new(Vec::new()));
        let provider = HybridProvider::new(
            "lmstudio".into(),
            "qwen-edge".into(),
            Box::new(MockProvider::new(
                vec!["edge answer"],
                edge_seen.clone(),
                false,
                false,
            )),
            "openrouter".into(),
            "claude-frontier".into(),
            Box::new(MockProvider::new(
                vec!["frontier answer"],
                frontier_seen.clone(),
                true,
                true,
            )),
            HybridRouteTarget::Frontier,
        );

        let response = provider
            .chat(
                ChatRequest {
                    messages: &[ChatMessage::user("what time is it?")],
                    tools: None,
                },
                "hybrid:auto",
                0.2,
            )
            .await
            .expect("hybrid request should succeed");

        assert_eq!(response.text.as_deref(), Some("frontier answer"));
        assert!(edge_seen.lock().is_empty());
        assert_eq!(frontier_seen.lock().as_slice(), &["claude-frontier"]);
    }

    #[tokio::test]
    async fn explicit_model_override_uses_selected_target() {
        let edge_seen = Arc::new(Mutex::new(Vec::new()));
        let frontier_seen = Arc::new(Mutex::new(Vec::new()));
        let provider = HybridProvider::new(
            "lmstudio".into(),
            "qwen-edge".into(),
            Box::new(MockProvider::new(
                vec!["edge answer"],
                edge_seen.clone(),
                false,
                false,
            )),
            "openrouter".into(),
            "claude-frontier".into(),
            Box::new(MockProvider::new(
                vec!["frontier answer"],
                frontier_seen.clone(),
                true,
                true,
            )),
            HybridRouteTarget::Frontier,
        );

        let response = provider
            .chat(
                ChatRequest {
                    messages: &[ChatMessage::user("use the fast path")],
                    tools: None,
                },
                "hybrid:edge",
                0.2,
            )
            .await
            .expect("hybrid request should succeed");

        assert_eq!(response.text.as_deref(), Some("edge answer"));
        assert_eq!(edge_seen.lock().as_slice(), &["qwen-edge"]);
        assert!(frontier_seen.lock().is_empty());
    }

    #[test]
    fn resolves_router_target_from_edge_by_default() {
        let hybrid = HybridProviderConfig {
            enabled: true,
            edge: HybridTargetConfig {
                provider: "lmstudio".into(),
                model: "qwen-edge".into(),
                api_key: None,
                api_url: Some("http://localhost:1234/v1".into()),
            },
            frontier: HybridTargetConfig {
                provider: "openrouter".into(),
                model: "claude-frontier".into(),
                api_key: None,
                api_url: None,
            },
            classifier: Default::default(),
        };

        let router = resolve_router_target(&RouterConfig::default(), &hybrid, Some("global-key"))
            .expect("router target should resolve");

        assert_eq!(router.provider_name, "lmstudio");
        assert_eq!(router.model, "qwen-edge");
        assert_eq!(router.api_key.as_deref(), Some("global-key"));
        assert_eq!(router.api_url.as_deref(), Some("http://localhost:1234/v1"));
    }
}
