"""
vault.py - encrypted local storage (Fernet, PRIVLAB_VAULT_KEY).

Holds the re-identification map per report, saved run state, learned names in the
profile and the Muse Spark cache. Nothing here is ever sent anywhere.
"""
import json

from cryptography.fernet import Fernet

from common.config import ROOT, load_config

VAULT_DIR = ROOT / "vault"


def fernet() -> Fernet:
    settings = load_config()
    settings.require("vault_key")
    return Fernet(settings.vault_key.encode())


def encrypt_json(obj) -> bytes:
    return fernet().encrypt(json.dumps(obj, ensure_ascii=False).encode("utf-8"))


def decrypt_json(token: bytes):
    return json.loads(fernet().decrypt(token).decode("utf-8"))


def save_vault(deid_id: str, data: dict):
    VAULT_DIR.mkdir(exist_ok=True)
    path = VAULT_DIR / f"{deid_id}.enc"
    path.write_bytes(encrypt_json(data))
    return path


def load_vault(deid_id: str) -> dict:
    return decrypt_json((VAULT_DIR / f"{deid_id}.enc").read_bytes())
