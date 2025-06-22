import os
import sys
import asyncio
from dotenv import load_dotenv
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

class RiskSynthesizer:
    def __init__(self):
        load_dotenv()
        self.app_name = "risk_synthesizer"
        self.user_id = "synth_user"
        self.session_id_base = "risk_synthesis_session"
        self.session_service = InMemorySessionService()
        self.runner = None
        self.agent = self._create_agent()

    def _create_agent(self):
        return Agent(
            name="risk_synthesizer",
            model="gemini-2.0-flash",
            description="Agent that synthesizes arguments from aggressive, neutral, and safe risk analysts to generate a cohesive risk summary.",
            instruction="""
You are the Risk Synthesizer. Your job is to read the arguments from three analysts:
- The **Aggressive Analyst**, who pushes for high-reward, high-risk strategies.
- The **Neutral Analyst**, who promotes a balanced and moderate investment plan.
- The **Safe Analyst**, who advocates for conservative and low-risk strategies.

**Task:**
1. Concisely summarize each analyst’s viewpoint.
2. Highlight key points of agreement and disagreement.
3. Identify blind spots or biases in any of the positions.
4. Provide a final strategic synthesis that integrates their inputs or offers a reasoned conclusion favoring one stance.

Avoid formatting, and write in a professional, human conversational tone.
"""
        )

    async def setup_session(self, session_key):
        session_id = f"{self.session_id_base}_{session_key}"
        try:
            await self.session_service.create_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=session_id
            )
            self.runner = Runner(
                agent=self.agent,
                app_name=self.app_name,
                session_service=self.session_service
            )
            return session_id
        except Exception as e:
            print(f"❌ Session setup failed: {str(e)}")
            return None

    async def synthesize(self, ticker_symbol, aggressive_response, neutral_response, safe_response, trader_plan, history=""):
        session_key = hash((ticker_symbol, trader_plan)) % (10 ** 8)
        session_id = await self.setup_session(session_key)
        if not session_id:
            return "Synthesis could not be started due to session setup failure."

        prompt = f"""
You are summarizing an internal debate between investment risk analysts about a trader's plan:

Trader Plan: {trader_plan}

🔴 Aggressive Analyst said:
{aggressive_response}

⚪ Neutral Analyst said:
{neutral_response}

🟢 Safe Analyst said:
{safe_response}

🕓 Debate History:
{history}

Your job is to:
1. Briefly summarize each analyst's position.
2. Compare and contrast them.
3. Identify any flaws or oversights.
4. Suggest a synthesized or final actionable recommendation.
"""

        content = types.Content(
            role='user',
            parts=[types.Part(text=prompt)]
        )

        try:
            events = self.runner.run_async(
                user_id=self.user_id,
                session_id=session_id,
                new_message=content
            )
            async for event in events:
                if event.is_final_response():
                    return event.content.parts[0].text
            return "Synthesis completed but no final response was received."
        except Exception as e:
            print(f"❌ Synthesis error: {str(e)}")
            return f"Synthesis failed with error: {str(e)}"

    def get_synthesis(self, ticker_symbol, aggressive_response, neutral_response, safe_response, trader_plan, history=""):
        try:
            return asyncio.run(self.synthesize(
                ticker_symbol,
                aggressive_response,
                neutral_response,
                safe_response,
                trader_plan,
                history
            ))
        except Exception as e:
            return f"Synthesis execution failed: {str(e)}"

from aggresive_risk_debator import AggressiveRiskDebator
from conservative_risk_debator import SafeRiskAnalyst
from neutral_risk_debator import NeutralRiskAnalyst
from synthesizer import RiskSynthesizer

ticker = "AAPL"
trader_plan = "Buy 1000 shares ahead of earnings."
sentiment_report = "Investors excited about AI launch; slight concern over valuations."
news_report = "U.S.-China tensions rise; global markets uncertain."
fundamentals_report = "High revenue growth, negative free cash flow."

# Instantiate agents
aggressive = AggressiveRiskDebator()
safe = SafeRiskAnalyst()
neutral = NeutralRiskAnalyst()
synth = RiskSynthesizer()

# Collect debate responses
risky_response = aggressive.get_aggressive_debate(ticker, sentiment_report, news_report, fundamentals_report)
neutral_response = neutral.get_neutral_debate(ticker, sentiment_report, news_report, fundamentals_report, risky_response)
safe_response = safe.get_conservative_debate(ticker, sentiment_report, news_report, fundamentals_report, risky_response, neutral_response)

# Synthesize
synthesis_result = synth.get_synthesis(
    ticker_symbol=ticker,
    aggressive_response=risky_response,
    neutral_response=neutral_response,
    safe_response=safe_response,
    trader_plan=trader_plan,
    history="Aggressive led, Neutral raised balance, Safe highlighted risks."
)

print("\n📌 Final Synthesized Recommendation:\n", synthesis_result)
