from __future__ import annotations

from deploy.envkeys import PROVIDER_ENV, parse_dotenv, resolve_keys


def test_parse_dotenv_ignores_comments_and_strips(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# comment\n"
        "\n"
        'ANTHROPIC_API_KEY="sk-ant-xyz"\n'
        "export MOONSHOT_API_KEY = 'ms-123' \n",
        encoding="utf-8",
    )
    parsed = parse_dotenv(p)
    assert parsed["ANTHROPIC_API_KEY"] == "sk-ant-xyz"
    assert parsed["MOONSHOT_API_KEY"] == "ms-123"


def test_resolve_keys_precedence_env_over_dotenv_over_file(tmp_path):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-dotenv\n", encoding="utf-8")
    (tmp_path / "api_key.txt").write_text("from-file-anthropic\nfrom-file-kimi\n", encoding="utf-8")
    # env wins for anthropic; .env is absent for kimi so api_key.txt line 2 wins
    keys = resolve_keys(repo_root=tmp_path, env={"ANTHROPIC_API_KEY": "from-env"})
    assert keys["anthropic"] == "from-env"
    assert keys["kimi"] == "from-file-kimi"


def test_resolve_keys_absent_is_none(tmp_path):
    keys = resolve_keys(repo_root=tmp_path, env={})
    assert keys == {"anthropic": None, "kimi": None}
    assert set(PROVIDER_ENV) == {"anthropic", "kimi"}
