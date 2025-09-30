

#!/usr/bin/fish
set -e

# Auto-fetch eQuilibrator cache using Python (relaxed SSL, retries)
uv run --python 3.11 glycolysis/equilibrator_cache_fetch.py

# Strict run to ensure all ΔG′ come from eQuilibrator (no TOML fallback)
uv run --python 3.11 --with requests --with equilibrator-api \
  kegg_api.py \
  --no-kegg \
  --use-equilibrator \
  --strict-equilibrator \
  --kf-source baseline --baseline-kf 1.0 \
  --temperature 298.15 --ph 7.0 --ionic-strength 0.25 --pmg 3.0 \
  --format table \
  --write-toml params_kegg_sabio_eq.toml