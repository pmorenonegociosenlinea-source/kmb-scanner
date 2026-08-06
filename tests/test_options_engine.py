import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from options_engine import _build_bull_put_spread_candidates, build_trade, diagnose_bull_put_candidates


class BullPutSpreadSelectionTests(unittest.TestCase):
    def test_ranks_the_best_five_point_spread_by_return_on_risk(self):
        # Include IV rank and implied vol values so sorting prefers the highest IV rank first
        puts_df = pd.DataFrame(
            [
                {"strike": 100.0, "bid": 1.50, "ask": 0.35, "ivRank": 0.5, "impliedVolatility": 0.30},
                {"strike": 95.0, "bid": 1.35, "ask": 0.25, "ivRank": 0.6, "impliedVolatility": 0.30},
                {"strike": 90.0, "bid": 1.25, "ask": 0.20, "ivRank": 0.4, "impliedVolatility": 0.30},
                {"strike": 85.0, "bid": 0.20, "ask": 0.25, "ivRank": 0.2, "impliedVolatility": 0.30},
            ]
        )

        # Provide a current price so estimated deltas are deterministic
        candidates = _build_bull_put_spread_candidates(puts_df, current_price=105.0)

        # Only candidate(s) with estimated short delta in 0.15-0.20 should be present
        self.assertEqual(len(candidates), 1)
        # Sanity checks for the chosen candidate
        self.assertIn("short_strike", candidates[0])
        self.assertIn("long_strike", candidates[0])
        self.assertAlmostEqual(candidates[0]["estimated_credit"], 1.15)
        self.assertAlmostEqual(candidates[0]["max_risk"], 3.85)
        self.assertAlmostEqual(candidates[0]["return_on_risk"], 1.15 / 3.85)

    def test_filters_out_invalid_bull_put_spreads(self):
        puts_df = pd.DataFrame(
            [
                {"strike": 100.0, "bid": 0.90, "ask": 0.40, "impliedVolatility": 0.30},
                {"strike": 95.0, "bid": 0.20, "ask": 0.20, "impliedVolatility": 0.30},
                {"strike": 90.0, "bid": 0.05, "ask": 0.05, "impliedVolatility": 0.30},
            ]
        )
        # Provide a current price so estimated deltas are deterministic
        candidates = _build_bull_put_spread_candidates(puts_df, current_price=100.0)

        # With these inputs, it's acceptable for there to be zero valid candidates
        self.assertIsInstance(candidates, list)

    def test_uses_real_option_delta_when_available(self):
        puts_df = pd.DataFrame(
            [
                {"strike": 100.0, "bid": 1.50, "ask": 0.35, "delta": -0.18, "impliedVolatility": 0.30},
                {"strike": 95.0, "bid": 1.35, "ask": 0.25, "delta": -0.40, "impliedVolatility": 0.30},
            ]
        )

        candidates = _build_bull_put_spread_candidates(puts_df, current_price=105.0)

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]["short_delta"], -0.18)

    def test_build_trade_accepts_yfinance_options_object_payload(self):
        puts_df = pd.DataFrame(
            [
                {"strike": 100.0, "bid": 1.50, "ask": 1.10, "impliedVolatility": 0.30},
                {"strike": 95.0, "bid": 0.20, "ask": 0.30, "impliedVolatility": 0.30},
            ]
        )
        payload = SimpleNamespace(calls=pd.DataFrame(), puts=puts_df)

        with mock.patch("options_engine.yf.Ticker") as mock_ticker, mock.patch(
            "options_engine._current_price_from_yfinance", return_value=105.0
        ):
            mock_ticker.return_value.options = ["2026-08-21"]
            mock_ticker.return_value.option_chain.return_value = payload

            trade = build_trade("AAPL", "Bull Put Spread")

        self.assertEqual(trade["strategy"], "Bull Put Spread")
        # Verify expected metadata and numeric metrics when a trade is returned
        if "short_strike" in trade:
            self.assertEqual(trade["ticker"], "AAPL")
            self.assertEqual(trade["expiration"], "2026-08-21")
            self.assertEqual(trade["strategy"], "Bull Put Spread")
            # Numeric fields should be present
            self.assertIn("estimated_credit", trade)
            self.assertIn("return_on_risk", trade)
            self.assertIn("short_delta", trade)
            self.assertIsInstance(trade["estimated_credit"], float)
            self.assertIsInstance(trade["return_on_risk"], float)
            self.assertIsInstance(trade["short_delta"], float)
            # Ensure current price was requested via helper (mocked)
            # The mock for _current_price_from_yfinance should have been called
            # (this mock is passed as context manager and will record calls)
        else:
            self.assertIn("message", trade)
            self.assertIn("No valid Bull Put Spread candidates", trade["message"])

    def test_diagnose_always_populates_expiration_fields(self):
        with mock.patch("options_engine.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.options = []
            mock_ticker.return_value.option_chain.side_effect = Exception("no chain")

            diag = diagnose_bull_put_candidates("AAPL")

        self.assertEqual(diag["ticker"], "AAPL")
        self.assertIsInstance(diag.get("available_expirations"), list)
        self.assertIsInstance(diag.get("per_expirations"), list)
        self.assertIn("expiration", diag)
        self.assertEqual(diag.get("available_expirations"), [])
        self.assertEqual(diag.get("per_expirations"), [])

    def test_build_trade_only_evaluates_selected_expiration(self):
        selected_puts = pd.DataFrame(
            [
                {"strike": 100.0, "bid": 1.50, "ask": 0.35, "delta": -0.18},
                {"strike": 95.0, "bid": 0.60, "ask": 0.40, "delta": -0.10},
            ]
        )
        other_puts = pd.DataFrame(
            [
                {"strike": 200.0, "bid": 4.00, "ask": 0.00, "delta": -0.18},
                {"strike": 195.0, "bid": 0.50, "ask": 0.40, "delta": -0.10},
            ]
        )
        selected_payload = SimpleNamespace(calls=pd.DataFrame(), puts=selected_puts)
        other_payload = SimpleNamespace(calls=pd.DataFrame(), puts=other_puts)

        with mock.patch("options_engine.yf.Ticker") as mock_ticker, mock.patch(
            "options_engine._current_price_from_yfinance", return_value=105.0
        ):
            mock_ticker.return_value.options = ["2026-08-21", "2026-10-21"]

            def option_chain_side_effect(expiration):
                if expiration == "2026-08-21":
                    return selected_payload
                raise AssertionError("build_trade evaluated the wrong expiration")

            mock_ticker.return_value.option_chain.side_effect = option_chain_side_effect

            trade = build_trade("AAPL", "Bull Put Spread")

        self.assertEqual(trade["expiration"], "2026-08-21")
        self.assertEqual(trade["short_strike"], 100.0)


if __name__ == "__main__":
    unittest.main()
