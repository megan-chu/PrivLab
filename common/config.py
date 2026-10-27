"""
config.py - settings loaded from .env (never committed; see .env.example).

Missing values are only an error when something needs them, so pattern work
and scoring run without an API key.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

# Settings field -> environment variable
ENV_NAMES = {
    "meta_api_key": "META_API_KEY",
    "meta_api_base_url": "META_API_BASE_URL",
    "meta_model": "META_MODEL",
    "seal_key": "PRIVLAB_SEAL_KEY",
    "vault_key": "PRIVLAB_VAULT_KEY",
}
DEFAULTS = {
    "meta_api_base_url": "https://api.meta.ai/v1",
    "meta_model": "muse-spark-1.3",
}


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    meta_api_key: str
    meta_api_base_url: str
    meta_model: str
    seal_key: str
    vault_key: str

    def require(self, *names: str) -> None:
        missing = [ENV_NAMES[n] for n in names if not getattr(self, n)]
        if missing:
            raise ConfigError(
                f"Missing in .env: {', '.join(missing)}. Copy .env.example to .env and fill them in."
            )


def load_config(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or ROOT / ".env", override=False)
    values = {
        field: os.environ.get(env, "").strip() or DEFAULTS.get(field, "")
        for field, env in ENV_NAMES.items()
    }
    if "-contributor" in values["meta_model"]:
        raise ConfigError(f"META_MODEL={values['meta_model']}: never use the -contributor variant.")
    return Settings(**values)
