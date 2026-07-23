//! Models runtime behavior contracts, ported from pi
//! `test/models-runtime.test.ts`.

use async_trait::async_trait;
use futures::future::BoxFuture;
use ri_llm_provider::auth::{
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthResult, AuthType, Credential,
    CredentialModifyFn, CredentialStore, InMemoryCredentialStore, ModelAuth, ModelsErrorCode,
    OAuthAuth, OAuthCredential, ProviderAuth, env_api_key_auth,
};
use ri_llm_provider::*;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn test_model(provider: &str, id: &str) -> Model {
    let mut model = Model::faux("faux", provider, id);
    model.base_url = "http://127.0.0.1:9".to_owned();
    model
}

fn oauth_credential(access: &str, expires: i64) -> Credential {
    Credential::OAuth(OAuthCredential {
        refresh: "refresh-token".to_owned(),
        access: access.to_owned(),
        expires,
        extra: Default::default(),
    })
}

struct StaticOAuth {
    refresh_calls: AtomicUsize,
    refresh_result: Result<(String, i64), String>,
}

impl StaticOAuth {
    fn succeeding(access: &str) -> Self {
        Self {
            refresh_calls: AtomicUsize::new(0),
            refresh_result: Ok((access.to_owned(), now_millis() + 3_600_000)),
        }
    }

    fn failing(error: &str) -> Self {
        Self {
            refresh_calls: AtomicUsize::new(0),
            refresh_result: Err(error.to_owned()),
        }
    }
}

#[async_trait]
impl OAuthAuth for StaticOAuth {
    fn name(&self) -> &str {
        "Test OAuth"
    }

    async fn login(
        &self,
        _interaction: &dyn ri_llm_provider::auth::AuthInteraction,
    ) -> Result<OAuthCredential, String> {
        Ok(OAuthCredential {
            refresh: "login-refresh".to_owned(),
            access: "login-access".to_owned(),
            expires: now_millis() + 3_600_000,
            extra: Default::default(),
        })
    }

    async fn refresh(&self, credential: &OAuthCredential) -> Result<OAuthCredential, String> {
        self.refresh_calls.fetch_add(1, Ordering::SeqCst);
        // Refresh under the lock is not instantaneous; give concurrent
        // resolvers a chance to race.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        match &self.refresh_result {
            Ok((access, expires)) => Ok(OAuthCredential {
                refresh: credential.refresh.clone(),
                access: access.clone(),
                expires: *expires,
                extra: credential.extra.clone(),
            }),
            Err(error) => Err(error.clone()),
        }
    }

    async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth, String> {
        Ok(ModelAuth {
            api_key: Some(credential.access.clone()),
            ..Default::default()
        })
    }
}

fn provider_with_auth(id: &str, auth: ProviderAuth, models: Vec<Model>) -> Arc<dyn Provider> {
    let handle = faux_provider(RegisterFauxProviderOptions {
        provider: Some(id.to_owned()),
        ..Default::default()
    });
    let mut options =
        CreateProviderOptions::new(id, auth, ProviderApiDispatch::Single(handle_api(&handle)));
    options.models = models;
    create_provider(options)
}

fn handle_api(handle: &FauxProviderHandle) -> Arc<dyn ApiProvider> {
    struct Passthrough {
        inner: Arc<dyn Provider>,
        api: String,
    }
    impl ApiProvider for Passthrough {
        fn api(&self) -> &str {
            &self.api
        }
        fn stream(
            &self,
            model: &Model,
            context: Context,
            options: StreamOptions,
        ) -> Result<AssistantMessageEventStream, ProviderError> {
            Ok(self.inner.stream(model, context, options))
        }
        fn stream_simple(
            &self,
            model: &Model,
            context: Context,
            options: SimpleStreamOptions,
        ) -> Result<AssistantMessageEventStream, ProviderError> {
            Ok(self.inner.stream_simple(model, context, options))
        }
    }
    Arc::new(Passthrough {
        inner: handle.provider.clone(),
        api: handle.api.clone(),
    })
}

fn always_configured_api_key(key: &str) -> Arc<dyn ApiKeyAuth> {
    struct Fixed {
        key: String,
    }
    #[async_trait]
    impl ApiKeyAuth for Fixed {
        fn name(&self) -> &str {
            "Fixed key"
        }
        async fn resolve(
            &self,
            _ctx: &dyn AuthContext,
            credential: Option<&ApiKeyCredential>,
        ) -> Result<Option<AuthResult>, String> {
            let key = credential
                .and_then(|credential| credential.key.clone())
                .unwrap_or_else(|| self.key.clone());
            Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key),
                    ..Default::default()
                },
                env: credential.map(|c| c.env.clone()).unwrap_or_default(),
                source: Some("fixed".to_owned()),
            }))
        }
    }
    Arc::new(Fixed {
        key: key.to_owned(),
    })
}

#[tokio::test]
async fn registers_replaces_and_deletes_providers() {
    let models = create_models(CreateModelsOptions::default());
    let p1 = provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: Some(always_configured_api_key("k")),
            oauth: None,
        },
        vec![test_model("p1", "m1")],
    );
    models.set_provider(p1);
    assert_eq!(models.get_providers().len(), 1);
    assert_eq!(models.get_models(Some("p1")).len(), 1);

    let replacement = provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: Some(always_configured_api_key("k")),
            oauth: None,
        },
        vec![test_model("p1", "m2"), test_model("p1", "m3")],
    );
    models.set_provider(replacement);
    assert_eq!(models.get_providers().len(), 1);
    assert_eq!(models.get_models(Some("p1")).len(), 2);
    assert!(models.get_model("p1", "m2").is_some());
    assert!(models.get_model("p1", "m1").is_none());

    models.delete_provider("p1");
    assert!(models.get_providers().is_empty());
    assert!(models.get_models(None).is_empty());
}

#[tokio::test]
async fn enumerates_credential_metadata_without_exposing_secrets() {
    let store = Arc::new(InMemoryCredentialStore::new());
    store
        .modify(
            "anthropic",
            Box::new(|_| Box::pin(async { Ok(Some(oauth_credential("secret", 0))) })),
        )
        .await
        .expect("store oauth");
    store
        .modify(
            "openai",
            Box::new(|_| {
                Box::pin(async {
                    Ok(Some(Credential::ApiKey(ApiKeyCredential {
                        key: Some("sk-test".to_owned()),
                        env: Default::default(),
                    })))
                })
            }),
        )
        .await
        .expect("store api key");

    let listed = store.list().await.expect("list");
    assert_eq!(listed.len(), 2);
    assert!(
        listed
            .iter()
            .any(|info| info.provider_id == "anthropic" && info.auth_type == AuthType::OAuth)
    );
    assert!(
        listed
            .iter()
            .any(|info| info.provider_id == "openai" && info.auth_type == AuthType::ApiKey)
    );
}

#[tokio::test]
async fn stored_credential_owns_provider_and_ambient_only_when_nothing_stored() {
    let store = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(store.clone()),
        ..Default::default()
    });
    let oauth = Arc::new(StaticOAuth::succeeding("refreshed-access"));
    models.set_provider(provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: Some(env_api_key_auth("P1 key", ["RI_TEST_P1_KEY"])),
            oauth: Some(oauth.clone()),
        },
        vec![test_model("p1", "m1")],
    ));

    // Nothing stored and no env: unconfigured.
    let auth = models
        .get_auth("p1", &Default::default())
        .await
        .expect("auth");
    assert!(auth.is_none());

    // Stored api-key credential wins.
    store
        .modify(
            "p1",
            Box::new(|_| {
                Box::pin(async {
                    Ok(Some(Credential::ApiKey(ApiKeyCredential {
                        key: Some("stored-key".to_owned()),
                        env: BTreeMap::from([(
                            "CLOUDFLARE_ACCOUNT_ID".to_owned(),
                            "acc".to_owned(),
                        )]),
                    })))
                })
            }),
        )
        .await
        .expect("store");
    let auth = models
        .get_auth("p1", &Default::default())
        .await
        .expect("auth")
        .expect("configured");
    assert_eq!(auth.auth.api_key.as_deref(), Some("stored-key"));
    assert_eq!(
        auth.env.get("CLOUDFLARE_ACCOUNT_ID").map(String::as_str),
        Some("acc")
    );
    assert_eq!(auth.source.as_deref(), Some("stored credential"));

    // A valid stored OAuth credential resolves without refreshing.
    store
        .modify(
            "p1",
            Box::new(|_| {
                Box::pin(async {
                    Ok(Some(oauth_credential(
                        "valid-access",
                        now_millis() + 60_000,
                    )))
                })
            }),
        )
        .await
        .expect("store oauth");
    let auth = models
        .get_auth("p1", &Default::default())
        .await
        .expect("auth")
        .expect("configured");
    assert_eq!(auth.auth.api_key.as_deref(), Some("valid-access"));
    assert_eq!(auth.source.as_deref(), Some("OAuth"));
    assert_eq!(oauth.refresh_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn expired_oauth_refreshes_once_and_persists_rotated_credential() {
    let store = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(store.clone()),
        ..Default::default()
    });
    let oauth = Arc::new(StaticOAuth::succeeding("rotated-access"));
    models.set_provider(provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: None,
            oauth: Some(oauth.clone()),
        },
        vec![test_model("p1", "m1")],
    ));
    store
        .modify(
            "p1",
            Box::new(|_| Box::pin(async { Ok(Some(oauth_credential("expired", 1))) })),
        )
        .await
        .expect("store");

    // Concurrent resolutions serialize through store.modify: one refresh.
    let overrides = ri_llm_provider::auth::AuthResolutionOverrides::default();
    let (first, second) = tokio::join!(
        models.get_auth("p1", &overrides),
        models.get_auth("p1", &overrides),
    );
    for auth in [first.expect("first"), second.expect("second")] {
        assert_eq!(
            auth.expect("configured").auth.api_key.as_deref(),
            Some("rotated-access")
        );
    }
    assert_eq!(oauth.refresh_calls.load(Ordering::SeqCst), 1);

    let stored = store.read("p1").await.expect("read").expect("stored");
    let Credential::OAuth(stored) = stored else {
        panic!("expected oauth credential");
    };
    assert_eq!(stored.access, "rotated-access");
    assert!(stored.expires > now_millis());
}

#[tokio::test]
async fn failed_oauth_refresh_rejects_with_oauth_code_and_preserves_credential() {
    let store = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(store.clone()),
        ..Default::default()
    });
    models.set_provider(provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: Some(always_configured_api_key("ambient-key")),
            oauth: Some(Arc::new(StaticOAuth::failing("invalid_grant"))),
        },
        vec![test_model("p1", "m1")],
    ));
    store
        .modify(
            "p1",
            Box::new(|_| Box::pin(async { Ok(Some(oauth_credential("expired", 1))) })),
        )
        .await
        .expect("store");

    let error = models
        .get_auth("p1", &Default::default())
        .await
        .expect_err("refresh must fail");
    assert_eq!(error.code, ModelsErrorCode::OAuth);
    assert!(error.to_string().contains("OAuth refresh failed for p1"));

    // No silent env fallback, and the stored credential is preserved for
    // retry/re-login.
    let stored = store.read("p1").await.expect("read").expect("stored");
    let Credential::OAuth(stored) = stored else {
        panic!("expected oauth credential");
    };
    assert_eq!(stored.access, "expired");
}

#[tokio::test]
async fn stored_credential_without_matching_handler_blocks_ambient_fallback() {
    let store = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(store.clone()),
        ..Default::default()
    });
    models.set_provider(provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: Some(always_configured_api_key("ambient-key")),
            oauth: None, // no OAuth handler
        },
        vec![test_model("p1", "m1")],
    ));
    store
        .modify(
            "p1",
            Box::new(|_| {
                Box::pin(async { Ok(Some(oauth_credential("token", now_millis() + 60_000))) })
            }),
        )
        .await
        .expect("store");

    let auth = models
        .get_auth("p1", &Default::default())
        .await
        .expect("auth");
    assert!(auth.is_none());
}

#[tokio::test]
async fn credential_store_failures_wrap_in_models_error() {
    struct FailingStore;
    #[async_trait]
    impl CredentialStore for FailingStore {
        async fn read(&self, _provider_id: &str) -> Result<Option<Credential>, String> {
            Err("disk exploded".to_owned())
        }
        async fn list(&self) -> Result<Vec<ri_llm_provider::auth::CredentialInfo>, String> {
            Err("disk exploded".to_owned())
        }
        async fn modify(
            &self,
            _provider_id: &str,
            _f: CredentialModifyFn<'_>,
        ) -> Result<Option<Credential>, String> {
            Err("disk exploded".to_owned())
        }
        async fn delete(&self, _provider_id: &str) -> Result<(), String> {
            Err("disk exploded".to_owned())
        }
    }

    let models = create_models(CreateModelsOptions {
        credentials: Some(Arc::new(FailingStore)),
        ..Default::default()
    });
    models.set_provider(provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: Some(always_configured_api_key("key")),
            oauth: None,
        },
        vec![test_model("p1", "m1")],
    ));

    let error = models
        .get_auth("p1", &Default::default())
        .await
        .expect_err("store read must fail");
    assert_eq!(error.code, ModelsErrorCode::Auth);
    assert!(error.to_string().contains("Credential store read failed"));

    let error = models.logout("p1").await.expect_err("delete must fail");
    assert_eq!(error.code, ModelsErrorCode::Auth);
}

#[tokio::test]
async fn api_key_auth_failures_wrap_in_models_error() {
    struct ExplodingAuth;
    #[async_trait]
    impl ApiKeyAuth for ExplodingAuth {
        fn name(&self) -> &str {
            "Exploding"
        }
        async fn resolve(
            &self,
            _ctx: &dyn AuthContext,
            _credential: Option<&ApiKeyCredential>,
        ) -> Result<Option<AuthResult>, String> {
            Err("resolver crashed".to_owned())
        }
    }

    let models = create_models(CreateModelsOptions::default());
    models.set_provider(provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: Some(Arc::new(ExplodingAuth)),
            oauth: None,
        },
        vec![test_model("p1", "m1")],
    ));
    let error = models
        .get_auth("p1", &Default::default())
        .await
        .expect_err("must fail");
    assert_eq!(error.code, ModelsErrorCode::Auth);
    assert!(error.to_string().contains("API key auth failed"));
}

#[tokio::test]
async fn explicit_request_api_key_and_env_win_during_resolution() {
    let models = create_models(CreateModelsOptions::default());
    models.set_provider(provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: Some(always_configured_api_key("default-key")),
            oauth: None,
        },
        vec![test_model("p1", "m1")],
    ));

    let overrides = ri_llm_provider::auth::AuthResolutionOverrides {
        api_key: Some("override-key".to_owned()),
        env: BTreeMap::from([("SCOPED".to_owned(), "yes".to_owned())]),
    };
    let auth = models
        .get_auth("p1", &overrides)
        .await
        .expect("auth")
        .expect("configured");
    assert_eq!(auth.auth.api_key.as_deref(), Some("override-key"));
    assert_eq!(auth.env.get("SCOPED").map(String::as_str), Some("yes"));
}

#[tokio::test]
async fn login_and_logout_run_through_the_credential_store() {
    struct NoInteraction;
    #[async_trait]
    impl ri_llm_provider::auth::AuthInteraction for NoInteraction {
        async fn prompt(
            &self,
            _prompt: ri_llm_provider::auth::AuthPrompt,
        ) -> Result<String, String> {
            Ok("prompted-key".to_owned())
        }
        fn notify(&self, _event: ri_llm_provider::auth::AuthEvent) {}
    }

    let store = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(store.clone()),
        ..Default::default()
    });
    models.set_provider(provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: Some(env_api_key_auth("P1 key", ["RI_TEST_P1_LOGIN_KEY"])),
            oauth: Some(Arc::new(StaticOAuth::succeeding("access"))),
        },
        vec![test_model("p1", "m1")],
    ));

    let credential = models
        .login("p1", AuthType::ApiKey, &NoInteraction)
        .await
        .expect("api key login");
    assert!(matches!(
        credential,
        Credential::ApiKey(ApiKeyCredential { ref key, .. }) if key.as_deref() == Some("prompted-key")
    ));
    assert!(store.read("p1").await.expect("read").is_some());

    let credential = models
        .login("p1", AuthType::OAuth, &NoInteraction)
        .await
        .expect("oauth login");
    assert!(matches!(credential, Credential::OAuth(_)));

    models.logout("p1").await.expect("logout");
    assert!(store.read("p1").await.expect("read").is_none());

    let error = models
        .login("missing", AuthType::ApiKey, &NoInteraction)
        .await
        .expect_err("unknown provider");
    assert_eq!(error.code, ModelsErrorCode::Provider);
}

#[tokio::test]
async fn check_auth_reports_without_refreshing_oauth_and_get_available_filters() {
    let store = Arc::new(InMemoryCredentialStore::new());
    let models = create_models(CreateModelsOptions {
        credentials: Some(store.clone()),
        ..Default::default()
    });
    let oauth = Arc::new(StaticOAuth::succeeding("access"));
    let filter_calls = Arc::new(AtomicUsize::new(0));
    let filter_calls_ref = filter_calls.clone();
    let handle = faux_provider(RegisterFauxProviderOptions {
        provider: Some("p1".to_owned()),
        ..Default::default()
    });
    let mut options = CreateProviderOptions::new(
        "p1",
        ProviderAuth {
            api_key: None,
            oauth: Some(oauth.clone()),
        },
        ProviderApiDispatch::Single(handle_api(&handle)),
    );
    options.models = vec![test_model("p1", "m1"), test_model("p1", "m2")];
    options.filter_models = Some(Arc::new(move |models, _credential| {
        filter_calls_ref.fetch_add(1, Ordering::SeqCst);
        models
            .into_iter()
            .filter(|model| model.id == "m1")
            .collect()
    }));
    models.set_provider(create_provider(options));

    // Unconfigured: no check result, no available models.
    assert!(models.check_auth("p1").await.expect("check").is_none());
    assert!(
        models
            .get_available(None)
            .await
            .expect("available")
            .is_empty()
    );

    // Expired stored OAuth: check reports without refreshing.
    store
        .modify(
            "p1",
            Box::new(|_| Box::pin(async { Ok(Some(oauth_credential("expired", 1))) })),
        )
        .await
        .expect("store");
    let check = models
        .check_auth("p1")
        .await
        .expect("check")
        .expect("configured");
    assert_eq!(check.auth_type, AuthType::OAuth);
    assert_eq!(oauth.refresh_calls.load(Ordering::SeqCst), 0);

    let available = models.get_available(None).await.expect("available");
    assert_eq!(
        available
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["m1"]
    );
    assert!(filter_calls.load(Ordering::SeqCst) >= 1);
}

#[tokio::test]
async fn refresh_updates_dynamic_providers_and_reports_failures() {
    let models = create_models(CreateModelsOptions::default());
    let fetch_calls = Arc::new(AtomicUsize::new(0));
    let fetch_calls_ref = fetch_calls.clone();

    let handle = faux_provider(RegisterFauxProviderOptions {
        provider: Some("dynamic".to_owned()),
        ..Default::default()
    });
    let mut options = CreateProviderOptions::new(
        "dynamic",
        ProviderAuth {
            api_key: Some(always_configured_api_key("key")),
            oauth: None,
        },
        ProviderApiDispatch::Single(handle_api(&handle)),
    );
    options.models = vec![test_model("dynamic", "static-model")];
    options.fetch_models = Some(Arc::new(move |_context| {
        let fetch_calls = fetch_calls_ref.clone();
        Box::pin(async move {
            fetch_calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![test_model("dynamic", "fetched-model")])
        }) as BoxFuture<'static, Result<Vec<Model>, String>>
    }));
    models.set_provider(create_provider(options));

    // Failing dynamic provider reports its error without failing the refresh.
    let failing_handle = faux_provider(RegisterFauxProviderOptions {
        provider: Some("failing".to_owned()),
        ..Default::default()
    });
    let mut failing = CreateProviderOptions::new(
        "failing",
        ProviderAuth {
            api_key: Some(always_configured_api_key("key")),
            oauth: None,
        },
        ProviderApiDispatch::Single(handle_api(&failing_handle)),
    );
    failing.fetch_models = Some(Arc::new(|_context| {
        Box::pin(async { Err("catalog fetch failed".to_owned()) })
            as BoxFuture<'static, Result<Vec<Model>, String>>
    }));
    models.set_provider(create_provider(failing));

    // Unconfigured dynamic providers are skipped entirely.
    let skipped_handle = faux_provider(RegisterFauxProviderOptions {
        provider: Some("skipped".to_owned()),
        ..Default::default()
    });
    let mut skipped = CreateProviderOptions::new(
        "skipped",
        ProviderAuth {
            api_key: Some(env_api_key_auth("Skipped key", ["RI_TEST_SKIPPED_KEY"])),
            oauth: None,
        },
        ProviderApiDispatch::Single(handle_api(&skipped_handle)),
    );
    let skipped_calls = Arc::new(AtomicUsize::new(0));
    let skipped_calls_ref = skipped_calls.clone();
    skipped.fetch_models = Some(Arc::new(move |_context| {
        let calls = skipped_calls_ref.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Vec::new())
        }) as BoxFuture<'static, Result<Vec<Model>, String>>
    }));
    models.set_provider(create_provider(skipped));

    let result = models.refresh(ModelsRefreshOptions::default()).await;
    assert!(!result.aborted);
    assert_eq!(fetch_calls.load(Ordering::SeqCst), 1);
    assert_eq!(skipped_calls.load(Ordering::SeqCst), 0);
    assert_eq!(result.errors.len(), 1);
    let error = result.errors.get("failing").expect("failing error");
    assert_eq!(error.code, ModelsErrorCode::ModelSource);
    assert!(error.to_string().contains("catalog fetch failed"));

    // Dynamic overlay merged over the baseline.
    let ids = models
        .get_models(Some("dynamic"))
        .into_iter()
        .map(|model| model.id)
        .collect::<Vec<_>>();
    assert_eq!(ids, vec!["static-model", "fetched-model"]);
}

#[tokio::test]
async fn refresh_persists_catalogs_and_restores_without_network() {
    let store: Arc<dyn ModelsStore> = Arc::new(InMemoryModelsStore::new());
    let build_models = |store: Arc<dyn ModelsStore>| {
        let models = create_models(CreateModelsOptions {
            models_store: Some(store),
            ..Default::default()
        });
        let handle = faux_provider(RegisterFauxProviderOptions {
            provider: Some("dynamic".to_owned()),
            ..Default::default()
        });
        let mut options = CreateProviderOptions::new(
            "dynamic",
            ProviderAuth {
                api_key: Some(always_configured_api_key("key")),
                oauth: None,
            },
            ProviderApiDispatch::Single(handle_api(&handle)),
        );
        options.fetch_models = Some(Arc::new(|context| {
            Box::pin(async move {
                if !context.allow_network {
                    return Err("network disabled".to_owned());
                }
                Ok(vec![test_model("dynamic", "remote-model")])
            }) as BoxFuture<'static, Result<Vec<Model>, String>>
        }));
        models.set_provider(create_provider(options));
        models
    };

    let online = build_models(store.clone());
    let result = online.refresh(ModelsRefreshOptions::default()).await;
    assert!(result.errors.is_empty());
    assert_eq!(online.get_models(Some("dynamic")).len(), 1);

    // A fresh collection sharing the store restores offline.
    let offline = build_models(store);
    assert!(offline.get_models(Some("dynamic")).is_empty());
    let result = offline
        .refresh(ModelsRefreshOptions {
            allow_network: Some(false),
            ..Default::default()
        })
        .await;
    assert!(result.errors.is_empty());
    assert_eq!(
        offline
            .get_models(Some("dynamic"))
            .into_iter()
            .map(|model| model.id)
            .collect::<Vec<_>>(),
        vec!["remote-model"]
    );
}

#[tokio::test]
async fn merges_resolved_auth_into_stream_options_with_explicit_options_winning() {
    let models = create_models(CreateModelsOptions::default());
    let handle = faux_provider(RegisterFauxProviderOptions {
        provider: Some("p1".to_owned()),
        ..Default::default()
    });
    let seen_options: Arc<parking_lot::Mutex<Vec<SimpleStreamOptions>>> =
        Arc::new(parking_lot::Mutex::new(Vec::new()));
    let seen_ref = seen_options.clone();
    handle.set_responses(vec![faux_response_factory(move |_, options, _, _| {
        seen_ref.lock().push(options.clone());
        faux_assistant_message("ok", Default::default())
    })]);

    struct HeaderAuth;
    #[async_trait]
    impl ApiKeyAuth for HeaderAuth {
        fn name(&self) -> &str {
            "Header auth"
        }
        async fn resolve(
            &self,
            _ctx: &dyn AuthContext,
            _credential: Option<&ApiKeyCredential>,
        ) -> Result<Option<AuthResult>, String> {
            Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some("auth-key".to_owned()),
                    headers: BTreeMap::from([
                        ("X-Auth".to_owned(), "resolved".to_owned()),
                        ("X-Shared".to_owned(), "from-auth".to_owned()),
                    ]),
                    base_url: Some("http://auth.example".to_owned()),
                },
                env: BTreeMap::from([("PROVIDER_ENV".to_owned(), "resolved".to_owned())]),
                source: None,
            }))
        }
    }

    let mut options = CreateProviderOptions::new(
        "p1",
        ProviderAuth {
            api_key: Some(Arc::new(HeaderAuth)),
            oauth: None,
        },
        ProviderApiDispatch::Single(handle_api(&handle)),
    );
    let mut model = handle.get_model();
    model.provider = "p1".to_owned();
    model
        .headers
        .insert("X-Model".to_owned(), "model-header".to_owned());
    options.models = vec![model.clone()];
    models.set_provider(create_provider(options));

    let mut stream_options = SimpleStreamOptions::default();
    stream_options
        .stream
        .headers
        .insert("x-shared".to_owned(), "explicit".to_owned());
    stream_options
        .stream
        .env
        .insert("REQUEST_ENV".to_owned(), "explicit".to_owned());
    let message = models
        .complete_simple(&model, user_context("hello"), stream_options)
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);

    let seen = seen_options.lock().clone();
    assert_eq!(seen.len(), 1);
    let seen = &seen[0];
    assert_eq!(seen.stream.api_key.as_deref(), Some("auth-key"));
    // Case-insensitive merge: the explicit header replaces the resolved one.
    assert_eq!(
        seen.stream.headers.get("x-shared").map(String::as_str),
        Some("explicit")
    );
    assert!(!seen.stream.headers.contains_key("X-Shared"));
    assert_eq!(
        seen.stream.headers.get("X-Auth").map(String::as_str),
        Some("resolved")
    );
    // Model headers merged through model-scoped auth resolution.
    assert_eq!(
        seen.stream.headers.get("X-Model").map(String::as_str),
        Some("model-header")
    );
    // Provider env and request env merge; request wins per key.
    assert_eq!(
        seen.stream.env.get("PROVIDER_ENV").map(String::as_str),
        Some("resolved")
    );
    assert_eq!(
        seen.stream.env.get("REQUEST_ENV").map(String::as_str),
        Some("explicit")
    );
}

#[tokio::test]
async fn unknown_providers_produce_error_streams_instead_of_panicking() {
    let models = create_models(CreateModelsOptions::default());
    let model = test_model("missing", "m1");
    let message = models
        .complete_simple(&model, user_context("hello"), Default::default())
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|error| error.contains("Unknown provider: missing"))
    );

    // Unconfigured providers also surface stream errors.
    let models = create_models(CreateModelsOptions::default());
    models.set_provider(provider_with_auth(
        "p1",
        ProviderAuth {
            api_key: Some(env_api_key_auth("P1 key", ["RI_TEST_P1_UNSET_KEY"])),
            oauth: None,
        },
        vec![test_model("p1", "m1")],
    ));
    let message = models
        .complete_simple(
            &test_model("p1", "m1"),
            user_context("hi"),
            Default::default(),
        )
        .await;
    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|error| error.contains("Provider is not configured: p1"))
    );
}

#[tokio::test]
async fn streams_through_the_provider_with_faux_handle() {
    let models = create_models(CreateModelsOptions::default());
    let handle = faux_provider(RegisterFauxProviderOptions::default());
    handle.set_responses(vec![
        faux_assistant_message("streamed", Default::default()).into(),
    ]);
    models.set_provider(handle.provider.clone());

    let model = handle.get_model();
    let message = models
        .complete_simple(&model, user_context("hello"), Default::default())
        .await;
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(
        message.content.first().and_then(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        }),
        Some("streamed")
    );
    assert_eq!(handle.state().call_count(), 1);
    assert!(has_api(&model, &handle.api));
}

#[tokio::test]
async fn builtin_models_exposes_catalog_providers() {
    let models = builtin_models(CreateModelsOptions::default());
    assert!(models.get_provider("anthropic").is_some());
    assert!(models.get_provider("openai").is_some());
    let model = models
        .get_model("anthropic", "claude-haiku-4-5")
        .expect("catalog model");
    assert_eq!(model.api, "anthropic-messages");

    // Built-in providers resolve auth through scoped env overrides.
    let overrides = ri_llm_provider::auth::AuthResolutionOverrides {
        api_key: Some("sk-explicit".to_owned()),
        env: Default::default(),
    };
    let auth = models
        .get_auth("anthropic", &overrides)
        .await
        .expect("auth")
        .expect("configured");
    assert_eq!(auth.auth.api_key.as_deref(), Some("sk-explicit"));
}

fn user_context(text: &str) -> Context {
    Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage::text(text))],
        tools: Vec::new(),
    }
}

#[tokio::test]
async fn resolves_stored_copilot_oauth_credentials_including_per_credential_base_url() {
    let store = Arc::new(InMemoryCredentialStore::new());
    let access = "tid=test;exp=9999999999;proxy-ep=proxy.enterprise.ghe.com;";
    store
        .modify(
            "github-copilot",
            Box::new(move |_| {
                Box::pin(async move {
                    Ok(Some(Credential::OAuth(OAuthCredential {
                        refresh: "ghu_refresh".to_owned(),
                        access: access.to_owned(),
                        expires: i64::MAX,
                        extra: serde_json::json!({ "enterpriseUrl": "ghe.example.com" })
                            .as_object()
                            .cloned()
                            .expect("extra"),
                    })))
                })
            }),
        )
        .await
        .expect("store credential");
    let models = create_models(CreateModelsOptions {
        credentials: Some(store),
        ..Default::default()
    });
    models.set_provider(builtin_provider("github-copilot").expect("copilot provider"));

    let resolution = models
        .get_auth("github-copilot", &Default::default())
        .await
        .expect("resolve")
        .expect("configured");
    assert_eq!(resolution.auth.api_key.as_deref(), Some(access));
    // The per-credential base URL derives from the token's proxy endpoint.
    assert_eq!(
        resolution.auth.base_url.as_deref(),
        Some("https://api.enterprise.ghe.com")
    );
}
