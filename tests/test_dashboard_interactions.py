import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_dashboard_runtime_asset_is_valid_javascript():
    js = ROOT / "src/scrarebench/assets/dashboard.js"
    assert js.exists()
    subprocess.run(["node", "--check", str(js)], check=True)
    text = js.read_text(encoding="utf-8")
    for token in ("Seed Stability", "Runs & Seeds", "Rare-cell Explorer", "Add report"):
        assert token in text
