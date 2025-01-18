import aiohttp
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def _create_indicator_prompt(self, indicators: Dict, market_data: Dict) -> str:
        return f"""You are a professional cryptocurrency trader and technical analyst. Analyze these indicators and provide actionable insights. Consider the relationships between different indicators, market context, and risk management.

Market Data:
{json.dumps(market_data, indent=2)}

Technical Analysis:
{json.dumps(indicators, indent=2)}

1. Evaluate each indicator's signal strength and reliability
2. Find confirmations and divergences between indicators
3. Assess the current market regime
4. Identify key price levels and potential areas of interest
5. Consider volume profile and market microstructure
6. Evaluate risk/reward ratios for potential trades

Format response as JSON with the following structure:
{{
    "indicator_analysis": {{
        "<indicator_name>": {{
            "signal": "LONG/SHORT/NEUTRAL",
            "confidence": 0-1,
            "reasoning": "explanation"
        }}
    }},
    "market_context": {{
        "trend": "BULLISH/BEARISH/RANGING",
        "volatility": "HIGH/MEDIUM/LOW",
        "key_levels": {{
            "support": [],
            "resistance": [],
            "significance": "explanation"
        }}
    }},
    "recommendation": {{
        "position": "LONG/SHORT/NEUTRAL",
        "confidence": 0-1,
        "entry_points": [],
        "stop_loss": [],
        "targets": [],
        "timeframe": "explanation",
        "key_risks": []
    }}
}}"""

    async def analyze_indicators(self, market_data: Dict, indicators: Dict) -> Dict:
        try:
            prompt = self._create_indicator_prompt(indicators, market_data)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.ollama_url,
                    json={
                        "model": "mistral",
                        "prompt": prompt,
                        "temperature": 0.7,
                        "stream": False
                    },
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '')
                        return self._parse_llm_response(response_text)
                    else:
                        logger.error(f"LLM request failed with status {response.status}")
                        return self._default_analysis()
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._default_analysis()

    def _parse_llm_response(self, response: str) -> Dict:
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            data = json.loads(response)
            
            # Validate required fields
            required_fields = ['indicator_analysis', 'market_context', 'recommendation']
            if not all(field in data for field in required_fields):
                logger.error("Missing required fields in LLM response")
                return self._default_analysis()
                
            return data
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._default_analysis()

    def _default_analysis(self) -> Dict:
        return {
            "indicator_analysis": {},
            "market_context": {
                "trend": "UNKNOWN",
                "volatility": "UNKNOWN",
                "key_levels": {
                    "support": [],
                    "resistance": [],
                    "significance": "Analysis failed"
                }
            },
            "recommendation": {
                "position": "NEUTRAL",
                "confidence": 0,
                "entry_points": [],
                "stop_loss": [],
                "targets": [],
                "timeframe": "Analysis failed",
                "key_risks": ["Analysis system failure"]
            }
        }