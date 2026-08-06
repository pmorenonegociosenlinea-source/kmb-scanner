import json
from data import TICKERS
from options_engine import debug_expirations_for_ticker

results = {}
for t in TICKERS:
    results[t] = debug_expirations_for_ticker(t)

print(json.dumps(results, indent=2))
