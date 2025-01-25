import aiohttp
import json
import logging
import torch
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class AnalysisEngine:
    def __init__(self, indicators, market_analyzer, ollama_url: str, device: torch.device):
        self.indicators = indicators
        self.market_analyzer = market_analyzer
        self.ollama_url = ollama_url
        self.device = device
        self.models = {}
        self.scalers = {}

    def prepare_model_input(self, timeframe_data: Dict[str, pd.DataFrame], scaler) -> Optional[torch.Tensor]:
        try:
            if not timeframe_data:
                logger.debug("No timeframe data provided")
                return None

            features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macd_signal', 'vwap', 'atr']
            processed_data = []

            logger.debug(f"Processing timeframes: {list(timeframe_data.keys())}")
            for tf in sorted(timeframe_data.keys()):
                if timeframe_data[tf] is None or timeframe_data[tf].empty:
                    logger.debug(f"Empty or None DataFrame for timeframe {tf}")
                    continue

                df = timeframe_data[tf].copy()
                logger.debug(f"Timeframe {tf} data shape before indicators: {df.shape}")
                
                # Check for required columns
                missing_cols = [col for col in features if col not in df.columns]
                if missing_cols:
                    logger.debug(f"Missing columns for {tf}: {missing_cols}")
                    indicators_df = self.indicators.calculate_all_timeframes({tf: df})[tf]
                    df = indicators_df.copy()
                    logger.debug(f"Added indicators. New shape: {df.shape}")

                last_window = df[features].tail(12).values
                logger.debug(f"Window shape for {tf}: {last_window.shape}")

                if scaler:
                    last_window = scaler.transform(last_window)
                    logger.debug(f"Scaled window shape: {last_window.shape}")

                processed_data.append(last_window)

            if not processed_data:
                logger.debug("No processed data available")
                return None

            tensor_input = torch.tensor(processed_data, dtype=torch.float32).unsqueeze(0)
            logger.debug(f"Final tensor shape: {tensor_input.shape}")
            return tensor_input

        except Exception as e:
            logger.error(f"Error preparing model input: {e}", exc_info=True)
            return None

    async def analyze_pair(self, pair: str, df: pd.DataFrame,
                           timeframe_data: Dict[str, pd.DataFrame],
                           portfolio_state: Dict) -> Dict:
        try:
            logger.debug(f"Analyzing pair: {pair}")
            indicators = self.indicators.calculate_all(df)
            regime_data = self.market_analyzer.analyze_regime(df)
            
            logger.debug(f"Checking for model and scaler: {pair}")
            logger.debug(f"Available models: {list(self.models.keys())}")
            logger.debug(f"Available scalers: {list(self.scalers.keys())}")
            
            ml_predictions = await self._get_ml_predictions(pair, timeframe_data)
            if ml_predictions is not None:
                logger.debug(f"ML predictions shape: {ml_predictions.shape}")
            else:
                logger.debug("No ML predictions returned")

            llm_decision = await self.get_llm_analysis(
                df=df,
                indicators=indicators,
                regime_data=regime_data,
                pair=pair,
                portfolio_state=portfolio_state,
                ml_predictions=ml_predictions
            )

            return {
                'indicators': indicators,
                'regime_data': regime_data,
                'ml_predictions': ml_predictions,
                'llm_decision': llm_decision
            }

        except Exception as e:
            logger.error(f"Error analyzing {pair}: {e}", exc_info=True)
            return self._default_analysis()

    async def _get_ml_predictions(self, pair: str, timeframe_data: Dict) -> Optional[torch.Tensor]:
        try:
            if pair not in self.models:
                logger.debug(f"No model found for {pair}")
                return None
            if pair not in self.scalers:
                logger.debug(f"No scaler found for {pair}")
                return None

            logger.debug(f"Preparing model input for {pair}")
            model_input = self.prepare_model_input(timeframe_data, self.scalers[pair])
            if model_input is None:
                logger.debug("Model input preparation failed")
                return None

            logger.debug("Running model inference")
            with torch.no_grad():
                predictions = self.models[pair](model_input.to(self.device))
                logger.debug(f"Predictions shape: {predictions.shape}")
                logger.debug(f"Predictions values: {predictions.cpu().numpy()}")
            return predictions

        except Exception as e:
            logger.error(f"Error getting ML predictions for {pair}: {e}", exc_info=True)
            return None

    async def get_llm_analysis(self, df: pd.DataFrame, indicators: Dict,
                               regime_data: Dict, pair: str,
                               portfolio_state: Dict,
                               ml_predictions: Optional[torch.Tensor] = None) -> Dict:
        try:
            current_price = float(df['close'].iloc[-1])
            position = portfolio_state.get('positions', {}).get(pair, {}).get('amount', 0)

            ml_analysis = "None"
            if ml_predictions is not None:
                ml_analysis = (
                    f"The ML model output is: {ml_predictions.cpu().numpy().tolist()}. "
                    "Consider these predictions as a PRIMARY factor."
                )

            prompt = self._create_llm_prompt(
                pair=pair,
                current_price=current_price,
                df=df,
                indicators=indicators,
                regime_data=regime_data,
                position=position,
                portfolio_state=portfolio_state,
                ml_predictions=ml_predictions
            )

            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.post(
                        self.ollama_url,
                        json={
                            "model": "mistral",
                            "prompt": prompt,
                            "temperature": 0.2,
                            "stream": False
                        }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        raw_output = result.get('response', '')
                        try:
                            return json.loads(raw_output)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse LLM JSON: {e}")
                            return self._default_decision()
                    else:
                        logger.error(f"LLM request failed with status {response.status}")
                        return self._default_decision()

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}", exc_info=True)
            return self._default_decision()

    def _create_llm_prompt(self, pair: str, current_price: float, df: pd.DataFrame,
                           indicators: Dict, regime_data: Dict, position: float,
                           portfolio_state: Dict,
                           ml_predictions: Optional[torch.Tensor]) -> str:
        return f"""
You are a cryptocurrency trading strategist.

IMPORTANT: You must respond ONLY in valid JSON.  
Use EXACTLY this schema with no extra text outside:

{{
  "decision": {{
    "action": "BUY/SELL/HOLD",
    "size": "0.0-1.0",
    "stop_loss": "price level",
    "take_profit": "price level"
  }},
  "reasoning": {{
    "technical_analysis": "...",
    "risk_assessment": "...",
    "market_context": "...",
    "ml_analysis": "{ml_predictions.cpu().numpy().tolist() if ml_predictions is not None else 'No ML predictions available'}"
  }}
}}

Now analyze:

Ticker Pair: {pair}

Market Data:
- Current Price: ${current_price:.4f}
- 24h Change: {float(df['close'].pct_change(1440).iloc[-1] * 100):.2f}%
- Volume: {float(df['volume'].iloc[-1440:].sum()):.2f}

Technical Indicators:
{json.dumps(indicators, indent=2)}

Market Regime: 
{json.dumps(regime_data, indent=2)}

Portfolio Status:
- Current Position: {position}
- Available Balance: ${portfolio_state.get('usd_balance', 0):.2f}
"""

    def _default_decision(self) -> Dict:
        return {
            "decision": {
                "action": "HOLD",
                "size": 0.0,
                "stop_loss": None,
                "take_profit": None
            },
            "reasoning": {
                "technical_analysis": "Analysis error",
                "risk_assessment": "",
                "market_context": "",
                "ml_analysis": ""
            }
        }

    def _default_analysis(self) -> Dict:
        return {
            'indicators': {},
            'regime_data': {},
            'ml_predictions': None,
            'llm_decision': self._default_decision()
        }