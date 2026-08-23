"""Tests for OpenCode Free provider — registration, keyless contract, aliases.

The provider is KEYLESS: OpenCode's free tier is served anonymously and
rejects any unrecognized Authorization bearer with 401, so the provider
declares no env vars and every request goes out with an empty Authorization
header (see hermes_cli.models.opencode_zen_free_runtime).
"""

import os
from unittest.mock import patch


class TestOpenCodeFreeProviderRegistration:
    """Verify the opencode-free provider registers correctly."""

    def test_provider_is_registered(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile is not None
        assert profile.name == "opencode-free"

    def test_provider_has_correct_base_url(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile.base_url == "https://opencode.ai/zen/v1"

    def test_provider_is_keyless(self):
        """No env vars declared — the free tier requires no credential."""
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile.env_vars == ()

    def test_provider_headers_override_sdk_bearer(self):
        """The profile's default headers blank Authorization so the SDK's
        Bearer never reaches the wire (the free tier 401s unknown bearers)."""
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile.default_headers.get("Authorization") == ""

    def test_provider_uses_chat_completions_mode(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode-free")
        assert profile.api_mode == "chat_completions"


class TestOpenCodeFreeAliases:
    """Verify alias resolution for the opencode-free provider."""

    def test_alias_free(self):
        from providers import get_provider_profile
        profile = get_provider_profile("free")
        assert profile is not None
        assert profile.name == "opencode-free"

    def test_alias_opencode_free(self):
        from providers import get_provider_profile
        profile = get_provider_profile("opencode_free")
        assert profile is not None
        assert profile.name == "opencode-free"


class TestOpenCodeFreeAuthAlias:
    """Verify the hardcoded alias in auth.py resolve_provider()."""

    def test_resolve_provider_free_alias(self):
        from hermes_cli.auth import resolve_provider
        # "free" should resolve to "opencode-free" without any credential
        result = resolve_provider("free")
        assert result == "opencode-free"


class TestOpenCodeFreeModelLists:
    """Curated keyless model lists exist and stay in sync."""

    def test_fallback_models_exist(self):
        from hermes_cli.setup import _DEFAULT_PROVIDER_MODELS
        assert "opencode-free" in _DEFAULT_PROVIDER_MODELS

    def test_setup_list_matches_curated_catalog(self):
        """setup.py sample list must be a subset of the curated catalog
        (behavior contract, not a frozen snapshot)."""
        from hermes_cli.models import _PROVIDER_MODELS
        from hermes_cli.setup import _DEFAULT_PROVIDER_MODELS
        curated = set(_PROVIDER_MODELS["opencode-free"])
        assert set(_DEFAULT_PROVIDER_MODELS["opencode-free"]) <= curated

    def test_every_curated_model_is_keyless(self):
        """Every model in the opencode-free catalog must satisfy the keyless
        predicate — a paid slug here would route with no auth and 401."""
        from hermes_cli.models import _PROVIDER_MODELS, is_opencode_zen_free_model
        for mid in _PROVIDER_MODELS["opencode-free"]:
            assert is_opencode_zen_free_model(mid), mid

    def test_ox_alpha_is_listed(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "x-preview-f-free" in _PROVIDER_MODELS["opencode-free"]


class TestOpenCodeFreeRuntimeKeyless:
    """The runtime resolver pins every opencode-free model keyless."""

    def test_free_provider_any_model_routes_keyless(self):
        from hermes_cli.models import (
            OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER,
            opencode_zen_free_runtime,
        )
        rt = opencode_zen_free_runtime("opencode-free", "big-pickle")
        assert rt is not None
        assert rt["api_key"] == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER
        assert rt["base_url"] == "https://opencode.ai/zen/v1"
        assert rt["default_headers"]["Authorization"] == ""

    def test_free_provider_muse_routes_responses(self):
        """opencode-free inherits Zen's per-model endpoint routing."""
        from hermes_cli.models import opencode_zen_free_runtime
        rt = opencode_zen_free_runtime(
            "opencode-free", "muse-spark-1.2-contributor-free"
        )
        assert rt is not None
        assert rt["api_mode"] == "codex_responses"


class TestOxAlphaReasoningEffortPropagation:
    """ox-alpha reasoning_effort must reach the wire on every profile that
    serves it.

    The ox-alpha wire contract is low/high/max only (anything else 400s).
    Regression context: OpenCodeGoProfile.build_api_kwargs_extras was
    overridden (so the generic extra_body.reasoning fallback never fired)
    but had no ox-alpha branch, and _build_ox_alpha_reasoning_extras only
    matched x-preview-f-free — so a configured effort was silently dropped
    for ``ox-alpha-free`` on opencode-go.
    """

    def test_go_profile_high_passes_through(self):
        from providers import get_provider_profile

        profile = get_provider_profile("opencode-go")
        _, top_level = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="ox-alpha-free",
        )
        assert top_level == {"reasoning_effort": "high"}

    def test_go_profile_ultra_clamps_to_max(self):
        """ultra is Hermes-internal vocabulary; ox-alpha tops out at max."""
        from providers import get_provider_profile

        profile = get_provider_profile("opencode-go")
        _, top_level = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "ultra"},
            model="ox-alpha-free",
        )
        assert top_level == {"reasoning_effort": "max"}

    def test_go_profile_medium_clamps_down_to_low(self):
        """Clamping never escalates cost: medium → low on a 3-tier wire."""
        from providers import get_provider_profile

        profile = get_provider_profile("opencode-go")
        _, top_level = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="ox-alpha-free",
        )
        assert top_level == {"reasoning_effort": "low"}

    def test_disabled_reasoning_sends_nothing(self):
        from providers import get_provider_profile

        profile = get_provider_profile("opencode-go")
        extra_body, top_level = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="ox-alpha-free",
        )
        assert extra_body == {}
        assert top_level == {}

    def test_unrelated_model_still_gets_nothing(self):
        """Models without a declared contract keep server defaults."""
        from providers import get_provider_profile

        profile = get_provider_profile("opencode-go")
        extra_body, top_level = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="some-future-relay-model",
        )
        assert extra_body == {}
        assert top_level == {}

    def test_zen_free_profile_accepts_both_slugs(self):
        """x-preview-f-free and ox-alpha-free share one wire contract."""
        from plugins.model_providers.opencode_zen import (
            _build_ox_alpha_reasoning_extras,
        )

        for slug in ("x-preview-f-free", "ox-alpha-free"):
            _, top_level = _build_ox_alpha_reasoning_extras(
                {"enabled": True, "effort": "high"}, slug
            )
            assert top_level == {"reasoning_effort": "high"}, slug
