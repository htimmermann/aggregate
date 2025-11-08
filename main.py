import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal
import os
import ast
import re
from openai import OpenAI
import numpy as np

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)

class FallTemplateBot2025(ForecastBot):
    """
    This is a copy of the template bot for Fall 2025 Metaculus AI Tournament.
    """

    _max_concurrent_questions = (
        1 
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    DEFAULT_ENSEMBLE_MODEL_GROUPS = ["openai/gpt-oss-120b"] * 10


    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:

            #Comprehensive Research Prompt
            research_prompt = clean_indents(
                f"""
You are a research pre-processor for a forecasting system.

Your only task is to collect and structure all factual, recent, and historically relevant information that a forecasting model would need to answer the question. Do not make a forecast. Do not express likelihood. Do not omit recent events if they could change the base rate. Do not add filler.

The question is:
{question.question_text}

OUTPUT MUST FOLLOW THE EXACT SECTION ORDER BELOW.

============================================================
SECTIONS (IN THIS ORDER):

QUESTION:
Restate the question verbatim: {question.question_text};  This question's outcome will be determined by the specific criteria below: {question.resolution_criteria}, {question.fine_print}

QUESTION CLASSIFICATION:
Identify the main type(s) from this list (pick 1–3):
- policy_legislation_regulation (e.g. "Will the US Congress pass...", "Will the US require...", "Will FTC ban non-competes...")
- executive_agency_action (e.g. HHS/CDC/WHO declarations, OFAC, export controls, licensing)
- corporate_product_strategy (e.g. "Will Google implement...", "Will Apple allow...", "Will OpenAI offer...", "Will Boeing file for bankruptcy...")
- AI_safety_governance (e.g. joint statements, AGI claims, AI export controls, AI licensing)
- geopolitics_conflict_sanctions (e.g. nuclear tests, closures of straits, troop deaths, ceasefires)
- macro_financial_timeseries (e.g. “ending value of UST 10Y”, “Nvidia vs Apple returns”, “VIX intraday max”, “nonfarm payrolls”)
- elections_leadership (e.g. “Who will be president…”, “Will Netanyahu remain…”, “UK Labour leadership…”)
- health_outbreak_pandemic (e.g. H5N1 human-to-human, PHEIC, US public health emergency)
- climate_energy_IRA (e.g. 45X, 45Y, 48E requirements, IRA repeal/adder changes)
- philanthropy_global_dev (e.g. GiveWell, New Incentives, chlorine grants)
- space_launch_tech (e.g. SpaceX launch failures, orbital launches)
- sports_cultural_events (e.g. LoL Worlds, F1, NY Marathon times)

SUMMARY_OF_TARGET_EVENT:
1–4 sentences describing what the question is about, including deadline, jurisdiction, and main decision-maker(s). Include the exact deadline present in the question.

RECENT_DEVELOPMENTS:
List the most recent factual events, proposals, public statements, filings, press releases, or news reports that bear directly on the question. Include dates. Prefer the last 12–18 months. For policy questions, include: bill text introduced, committee actions, executive orders, court challenges, agency NPRMs, relevant elections. For corporate questions, include: official product announcements, regulatory pressure, EU/UK/US competition rulings, developer betas, and prior commitments. For macro/markets, include: last observed values and date stamps. Write as bullet points.

INSTITUTIONAL_AND_PROCEDURAL_CONSTRAINTS:
Describe the exact pathway by which the event in the question could occur, in that jurisdiction/organization.
Examples:
- US Congress: introduction → committee → chamber votes → reconciliation → president.
- EU/UK competition/DSA/DMA: investigation → preliminary finding → compliance deadline → appeal.
- US executive/agency: statutory authority → public health emergency criteria → precedent.
- Corporate: regulatory pressure (e.g. UK CMA, EU DMA) → compliance window → software change shipped.
- Geopolitics: military capability → political objective → escalation ladder → third-party mediation.
Be concrete about which bodies must act.

HISTORICAL_PRECEDENTS_AND_BASELINES:
Provide historical examples that are structurally similar to the question.
- For public health (e.g. “Will H5N1 get PHEIC?”): list past PHEICs, their triggers, case counts, geographic spread, and time from first detection to declaration.
- For tariff/trade/IRA changes: list prior uses of similar authorities, successful vs failed attempts, court challenges.
- For “Will X company do Y?” under regulatory pressure: list cases where Apple/Google/Meta changed product behavior in 1) EU, 2) UK, 3) US because of regulation.
- For “Will Strait of Hormuz be closed?”: list past partial disruptions, naval incidents, sanctions-linked escalations.
- For “Will UN have >193 members?”: list last admissions, criteria, current candidates.
If there is no close precedent, state “no close precedent; closest analogues are: …”.

KEY_ACTORS_AND_INCENTIVES:
List the decision-makers and their observable incentives.
Include:
- governments (US Congress, White House, agencies)
- foreign governments involved
- companies (Apple, Google, OpenAI, Nvidia, Boeing, SpaceX, Maersk…)
- regulators (FTC, CMA, EC, OFAC, USTR)
- multilateral orgs (UN, NATO, WHO, IMF, EU)
For each actor, state: role, lever of control, evidence (statement/action), and whether they can unilaterally cause the event.

DATA_AND_INDICATORS:
Provide hard data relevant to the question. Tailor by class:

- macro_financial_timeseries:
  - latest available value(s) for the instrument(s) in question (e.g. UST 10Y, ICE BofA HY OAS, VIX intraday high, S&P 500 futures, Nasdaq-100 futures, crude oil futures, gold futures) with dates.
  - recent volatility or spread patterns.
  - scheduled releases (CPI, payrolls, FOMC, earnings dates).
  - known seasonal patterns if the question is month-specific (e.g. Nov-25 payrolls).

- equities_earnings (e.g. “first reported EPS after Sep 2025 for TSLA/META/MSFT”):
  - last 4 quarters’ reported EPS/revenues with dates.
  - company’s reporting cadence and expected next report window.
  - major company guidance or known headwinds/tailwinds.
  - any corporate events that could affect revenue/EPS (product launches, regulatory fines, supply chain issues).

- health_outbreak_pandemic:
  - latest human and animal case counts, by country.
  - current CDC/WHO/HHS risk assessment levels.
  - confirmed or suspected human-to-human transmission events.
  - vaccination/antiviral stockpiles and funding.
  - geographic spread and biosecurity incidents.

- policy_legislation_regulation / IRA_energy:
  - current statutory text or proposed amendments (45X, 45Y, 48E etc.).
  - compliance or domestic content deadlines.
  - litigation or repeal attempts (identify chamber, bill name, sponsor).
  - relevant economic or sectoral data (domestic manufacturing capacity, import dependencies).

- geopolitics_conflict_sanctions:
  - current military activity, troop presence, recent casualties.
  - recent negotiations or peace talks (date, actors, outcome).
  - sanctions regimes in effect and recent escalations.
  - known trigger events for escalation (attacks on shipping, drone attacks, pipeline disruptions).

STRUCTURAL_OR_LONG-RUN_FACTORS:
Describe background factors that change slowly but affect the forecast:
- election calendars and likely partisan control shifts
- regulatory waves (AI safety, tech competition, export controls)
- ongoing wars and frozen conflicts
- climate trends relevant to weather/hurricane/temperature questions
- AI race dynamics (frontier labs, compute bottlenecks, GPU reporting regimes)
- IRA implementation dynamics (domestic content, adders, repeal attempts)

EDGE_CASES_AND_BLOCKERS:
List specific developments that would make the event much harder or impossible:
- adverse court ruling
- change in legislative majority
- company exiting a market
- treaty/vote requiring unanimity
- technological infeasibility within the time window
Also list “fast paths” (e.g. emergency authority, executive order, forced compliance via DMA/CMA).

REFERENCES:
List the sources or source types a researcher should pull from (exact agency/company/org names; add report names where relevant). Prefer:
- US: congress.gov, whitehouse.gov, federalregister.gov, treasury.gov, commerce.gov, hhs.gov, cdc.gov
- Multilateral: who.int, un.org, nato.int, imf.org, worldbank.org
- Corporate: investor relations pages, SEC/EDGAR, official press rooms
- Markets/data: Fed, BLS, BEA, EIA, ICE, CME
Do not describe the source, just list it.

============================================================
CONSTRAINTS:
- Do NOT make a forecast or say anything like “likely,” “unlikely,” “could,” “may,” “expected.”
- Do NOT argue or prioritize scenarios.
- Do NOT output explanations of what you are doing.
- Do NOT talk to the user; just output the sections.
- Be terse but complete.

                """
            )
            
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set in environment")

            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

            def get_forecast(model_name: str, message: str) -> str:
                completion = client.chat.completions.create(
                    extra_body={},
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": message,
                        }
                    ],
                )
                agent_response = completion.choices[0].message.content
                print(
                    "Model Used: "
                    + model_name
                    + " \n Reasoning: "
                    + agent_response
                    + "\n \n \n"
                )
                return agent_response
            
            research_report = get_forecast(model_name='openai/gpt-3.5-turbo', message=research_prompt)
           
            return research_report
        
    def make_forecast_prompt(self, question, research):
        base_forecast_prompt = f"""

            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question’s outcome will be determined by the following criteria (which have not yet been satisfied):
            {question.resolution_criteria}
            {question.fine_print}

            Research briefing:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Your task:
            - Reason step-by-step through the question using only the information provided in the research briefing.
            - Identify key causal mechanisms, dependencies, and constraints that would determine the outcome.
            - Consider both pathways in which critical events occur and those in which they do not — explain how each scenario would shape the outcome.
            - Be explicit about institutional timing, actor incentives, and structural barriers.
            - End with a short set of bullet points summarizing your reasoning chain and the main conditional factors that would change the answer
            """
        return base_forecast_prompt

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(self.make_forecast_prompt(question, research) +
            f"""
            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        agent_forecasts = []

        for agent in self.DEFAULT_ENSEMBLE_MODEL_GROUPS:

            reasoning = await self.get_llm("default", "llm").invoke(prompt)
            logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
            binary_prediction: BinaryPrediction = await structure_output(
                reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
            )
            decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

            logger.info(
                f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
            )

            agent_forecasts.append(ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning))

        logger.info(
                f"Total Agent Forecasts {agent_forecasts}"
        )

        return np.average(agent_forecasts)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(self.make_forecast_prompt(question, research) + 
            f"""
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}
            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(self.make_forecast_prompt(question, research) + 
            f"""

            Here are the lower and upper bounds:
            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            IMPORTANT: The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.nominal_upper_bound is not None:
            upper_bound_number = question.nominal_upper_bound
        else:
            upper_bound_number = question.upper_bound
        if question.nominal_lower_bound is not None:
            lower_bound_number = question.nominal_lower_bound
        else:
            lower_bound_number = question.lower_bound

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number}."
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number}."
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {lower_bound_number}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    assert run_mode in [
        "tournament",
        "metaculus_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=int(os.getenv("FORECAST_ENSEMBLE_SIZE", "3")),
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o-mini",  # "anthropic/claude-3-5-sonnet-20241022", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openrouter/openai/gpt-4o-mini",
            "researcher": "perplexity",
            "parser": "openrouter/openai/gpt-4o-mini",
            # Use GPT-5 on OpenRouter to generate personas (override via env or code)
            "persona_generator": "openrouter/openai/gpt-5",
        },
        # Optional: override via env FORECAST_OUTPUT_LIMIT
        output_limit=int(os.getenv("FORECAST_OUTPUT_LIMIT", "250")),
        prompt_variants=None,
    )

    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/"#,  # Human Extinction - Binary
            #"https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            #"https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            #"https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)
