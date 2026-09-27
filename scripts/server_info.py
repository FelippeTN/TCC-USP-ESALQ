"""Registra versão do vLLM e modelos servidos em cada endpoint, sem o endereço do servidor.

Uso, na raiz: py scripts/server_info.py [--out results/server_info_v3.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import EMBEDDING_MODEL, LLM_MODELS, SERVER_HOST


def probe(base_url: str) -> dict:
    info = {}
    for key, url in (("version", base_url.removesuffix("/v1") + "/version"), ("models", base_url + "/models")):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            info[key] = data.get("version") if key == "version" else [
                {k: m.get(k) for k in ("id", "root", "max_model_len")} for m in data.get("data", [])]
        except Exception as e:
            info[key + "_error"] = f"{type(e).__name__}: {e}".replace(SERVER_HOST, "<host>")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "server_info_v3.json")
    args = ap.parse_args()
    endpoints = {**{name: cfg["base_url"] for name, cfg in LLM_MODELS.items()}, "embedding": EMBEDDING_MODEL["base_url"]}
    report = {"collected_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "endpoints": {name: probe(url) for name, url in endpoints.items()}}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
