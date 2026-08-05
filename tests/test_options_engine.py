import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from options_engine import _build_bull_put_spread_candidates, build_trade


class BullPutSpreadSelectionTests(unittest.TestCase):
    def test_ranks_the_best_five_point_spread_by_return_on_risk(self):
        puts_df = pd.DataFrame(
            [
                {"strike": 100.0, "bid": 1.50, "ask": 0.35, "delta": -0.18},
                {"strike": 95.0, "bid": 1.35, "ask": 0.25, "delta": -0.17},
                {"strike": 90.0, "bid": 1.25, "ask": 0.20, "delta": -0.18},
                {"strike": 85.0, "bid": 0.20, "ask": 0.25, "delta": -0.45},
            ]
        )

        candidates = _build_bull_put_spread_candidates(puts_df)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["short_strike"], 100.0)
        self.assertEqual(candidates[0]["long_strike"], 95.0)
        self.assertAlmostEqual(candidates[0]["estimated_credit"], 1.25)
        self.assertAlmostEqual(candidates[0]["max_risk"], 3.75)
        self.assertAlmostEqual(candidates[0]["return_on_risk"], 1.25 / 3.75)
        self.assertAlmostEqual(candidates[0]["probability_of_profit"], 0.82)

    def test_filters_out_invalid_bull_put_spreads(self):
        puts_df = pd.DataFrame(
            [
                {"strike": 100.0, "bid": 0.90, "ask": 0.40, "delta": -0.18},
                {"strike": 95.0, "bid": 0.20, "ask": 0.20, "delta": -0.35},
                {"strike": 90.0, "bid": 0.05, "ask": 0.05, "delta": -0.45},
            ]
        )

        candidates = _build_bull_put_spread_candidates(puts_df)

        self.assertEqual(len(candidates), 0)

    def test_build_trade_accepts_yfinance_options_object_payload(self):
        puts_df = pd.DataFrame(
            [
                {"strike": 100.0, "bid": 1.50, "ask": 1.10, "delta": -0.20},
                {"strike": 95.0, "bid": 0.20, "ask": 0.30, "delta": -0.35},
            ]
        )
        payload = SimpleNamespace(calls=pd.DataFrame(), puts=puts_df)

        with mock.patch("options_engine.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.options = ["2026-08-21"]
            mock_ticker.return_value.option_chain.return_value = payload

            trade = build_trade("AAPL", "Bull Put Spread")

        self.assertEqual(trade["strategy"], "Bull Put Spread")
        self.assertEqual(trade["short_strike"], 100.0)
        self.assertEqual(trade["long_strike"], 95.0)
        self.assertEqual(trade["expiration"], "2026-08-21")
        self.assertAlmostEqual(trade["short_delta"], -0.2)
        self.assertAlmostEqual(trade["probability_of_profit"], 0.8)


if __name__ == "__main__":
    unittest.main()
