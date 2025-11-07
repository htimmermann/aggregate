import argparse
import asyncio
import logging
import os
import ast
import re
import json
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)


FALLBACK_PERSONA_HEADERS: list[str] = [
    (
        "You an expert leader-travel forecaster specialized in high-profile bilateral visits under legal, security, and political constraints. "
        "Consider: bilateral relations, election calendars, legal exposure/extradition risk, sanctions/visas, summit piggybacking, security assessments, public schedules/NOTAMs, prior travel patterns, logistics windows, media trial balloons, "
        "Show your reasoning limited to {output_limit} tokens; end with prediction as last tokens, on its own line"
    ),
    (
        "You an expert summit-attendance forecaster specialized in leader participation at G7/NATO/UN events from diplomatic and logistical signals. "
        "Consider: official agendas, domestic constraints, security/legal exposure, aircraft routing, proxy/ministerial substitutes, sanctions/visa issues, bilateral side-meetings, past attendance, health factors, last-minute cancellations, "
        "Show your reasoning limited to {output_limit} tokens; end with prediction as last tokens, on its own line"
    ),
    (
        "You an expert AI-benchmark forecaster specialized in leaderboard rank dynamics (e.g., Chatbot Arena) and #1 tenure. "
        "Consider: new model launches, evaluation settings, voting volume/brigading, rate limits, dataset familiarity, inference latency/cost, community sentiment, scoring rule changes, prior tenure, meta-updates, "
        "Show your reasoning limited to {output_limit} tokens; end with prediction as last tokens, on its own line"
    ),
]


# DEFAULT_ENSEMBLE_MODEL_GROUPS: dict[str, list[str]] = {
#     "n1": [
#         "openai/gpt-5",
#     ],
#     "n5": [
#         "anthropic/claude-sonnet-4.5",
#         "google/gemini-2.5-flash",
#         "x-ai/grok-4-fast",
#         "z-ai/glm-4.6",
#     ],
#     "n10": [
#         "openai/gpt-5-mini",
#         "anthropic/claude-sonnet-4",
#         "google/gemini-2.5-pro",
#         "deepseek/deepseek-chat-v3-0324",
#         "minimax/minimax-m2",
#     ],
#     "n25": [
#         "openai/gpt-oss-120b",
#         "openai/gpt-oss-20b",
#         "qwen/qwen3-235b-a22b-2507",
#         "google/gemini-2.0-flash-001",
#         "anthracite-org/magnum-v4-72b",
#         "meta-llama/llama-3.1-405b-instruct",
#         "z-ai/glm-4.5",
#         "z-ai/glm-4-32b",
#         "x-ai/grok-3",
#         "x-ai/grok-3-mini",
#         "amazon/nova-premier-v1",
#         "anthropic/claude-haiku-4.5",
#         "perplexity/sonar-pro",
#         "deepcogito/cogito-v2-preview-llama-405b",
#         "cohere/command-a",
#     ],
# }
DEFAULT_ENSEMBLE_MODEL_GROUPS = "openai/gpt-oss-120b"


def build_structured_research_prompt(question: str) -> str:
    return (
        f"""

You are a research pre-processor for a forecasting system.

Your only task is to collect and structure all factual, recent, and historically relevant information that a forecasting model would need to answer the question. Do not make a forecast. Do not express likelihood. Do not omit recent events if they could change the base rate. Do not add filler.

The question is:
{question}

OUTPUT MUST FOLLOW THE EXACT SECTION ORDER BELOW.

============================================================
SECTIONS (IN THIS ORDER):

QUESTION:
Restate the question verbatim: {question}

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
    ).strip()


def build_forecast_prompt(question: str, context: str) -> str:
    return (
        f"""
You are a forecasting model. You are given:
1. A forecasting question.
2. Factual, recent, structured research context prepared for you.

Your task:
- Give a single probability from 0 to 1 (not percent) that answers the question.
- Then give brief reasoning grounded ONLY in the provided context.
- Do NOT browse.
- Do NOT restate the entire context.
- Do NOT say "as an AI model".
- If the context is missing something, say what is missing.

RETURN YOUR ANSEWR IN JSON FORMAT ONLY AS FOLLOWS:
{
  "probability": <number between 0 and 1>,
  "reasoning": "<short explanation>"
}

QUESTION:
{question}

CONTEXT:
{context}
"""
    ).strip()


def _extract_json_block(text: str) -> str | None:
    if text is None:
        return None

    fenced = re.findall(r"```(?:json)?\\s*([\\s\\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        candidate = fenced[0].strip()
        if candidate:
            return candidate

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return None


def _normalize_probability(probability: float | int | str | None) -> float | None:
    if probability is None:
        return None

    if isinstance(probability, (int, float)):
        if 0 <= probability <= 1:
            return float(probability)
        if 1 < probability <= 100:
            return float(probability) / 100.0
        return None

    if isinstance(probability, str):
        candidate = probability.strip()
        if candidate.endswith("%"):
            try:
                return float(candidate[:-1]) / 100.0
            except ValueError:
                return None
        try:
            value = float(candidate)
        except ValueError:
            return None
        if 0 <= value <= 1:
            return value
        if 1 < value <= 100:
            return value / 100.0
    return None


def call_openrouter_model(client, model_name: str, question: str, context: str) -> dict:
    prompt = build_forecast_prompt(question, context)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response.choices[0].message.content if response.choices else ""
    probability = None
    reasoning = None

    json_text = _extract_json_block(content)
    if json_text:
        try:
            parsed = json.loads(json_text)
            probability = _normalize_probability(parsed.get("probability"))
            reasoning = parsed.get("reasoning")
        except Exception:
            reasoning = content
    else:
        reasoning = content

    usage = getattr(response, "usage", None)
    return {
        "model": model_name,
        "probability": probability,
        "reasoning": reasoning or content,
        "raw": content,
        "usage": usage,
    }


def run_openrouter_ensemble(
    client,
    model_groups: dict[str, list[str]],
    question: str,
    context: str,
) -> list[dict]:
    results: list[dict] = []
    for group_models in model_groups.values():
        for model in group_models:
            try:
                results.append(call_openrouter_model(client, model, question, context))
            except Exception as exc:  # pragma: no cover - network failure path
                logger.warning("OpenRouter request failed for %s: %s", model, exc)
    return results


def _aggregate_probabilities(results: list[dict]) -> float | None:
    probabilities = [r["probability"] for r in results if r.get("probability") is not None]
    if not probabilities:
        return None
    return sum(probabilities) / len(probabilities)


class FallTemplateBot2025(ForecastBot):
    """
    Forecast bot variant that mirrors the Metaculus template persona style while
    layering an OpenRouter ensemble pipeline for binary questions.
    """

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(
        self,
        *args,
        output_limit: int | None = None,
        prompt_variants: list[dict[str, str]] | None = None,
        ensemble_model_groups: dict[str, list[str]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.output_limit: int = (
            output_limit
            if output_limit is not None
            else int(os.getenv("FORECAST_OUTPUT_LIMIT", "250"))
        )
        self._personas_by_question: dict[str, list[tuple[str, str]]] = {}
        self._persona_cursor_by_question: dict[str, int] = {}
        self._structured_research_cache: dict[str, str] = {}
        self.ensemble_model_groups = ensemble_model_groups
        self._ensemble_results_by_question: dict[str, list[dict]] = {}
        self._openrouter_client = None
        self._warned_openrouter_failure = False
        self._prompt_variants = prompt_variants or []
        if self.ensemble_model_groups and not os.getenv("OPENROUTER_API_KEY"):
            logger.warning(
                "OPENROUTER_API_KEY is not set; disabling ensemble model groups."
            )
            self.ensemble_model_groups = None

    def _get_question_key(self, question: MetaculusQuestion) -> str:
        return (
            getattr(question, "page_url", None)
            or getattr(question, "question_text", None)
            or str(id(question))
        )

    async def _ensure_personas_for_question(self, question: MetaculusQuestion) -> None:
        key = self._get_question_key(question)
        if key in self._personas_by_question:
            return

        n = getattr(self, "predictions_per_research_report", None)
        if not isinstance(n, int) or n <= 0:
            n = int(os.getenv("FORECAST_ENSEMBLE_SIZE", "3"))

        question_text = question.question_text
        output_limit = self.output_limit
        prompt = (
            f"""
You are generating {n} expert forecaster PERSONAS for the single forecasting QUESTION below.

OBJECTIVE
Return a Python list of {n} strings. Each string must be formatted EXACTLY:
"You an expert <domain> forecaster specialized in <narrow specialization tied to the QUESTION>. Consider: <10 concise, comma-separated considerations>, Show your reasoning limited to {output_limit} tokens; end with prediction as last tokens, on its own line"

INPUT
QUESTION: <<{question_text}>>
OUTPUT_LIMIT: {output_limit}

RULES
1) Infer exactly ONE concise <domain> phrase from the QUESTION and use it verbatim in EVERY item. Do not mix domains.
   Examples of valid domain families (pick the closest one, or coin a concise equivalent): leader-travel; summit-attendance; AI-benchmark; podcast-chart; head-to-head podcast rank; Treasury-financing; US-macro releases nowcaster; state-GDP; spaceflight-operations; election-primaries; Euro-area sentiment; US consumer-sentiment; climate-anomaly; China inflation-cycle; rental-market; box-office; streaming-charts; geopolitical-bloc; diplomatic-recognition; West Africa border-policy; market-design; public-sector-efficiency; legal-outcome; IMF-program; fugitive-recapture; energy-program-policy; hydrology & drought; cross-country approval; candidacy-status; executive-clemency; party-alignment; web-availability; accords/treaty-signing; wealth-index composition; displacement-statistics; fuel-price nowcaster; legislative-throughput & confirmations; mortgage-rate; freight-transport; IPO-pipeline announcements; retail-platform policy; urban-complaints; UAP-reporting; air-service; asset-extremes; higher-ed metrics & compliance; social & media signal; live-events & awards; public-health surveillance; corporate-events.
2) Create {n} DISTINCT experts by varying sub-specialization/background and which considerations they emphasize (e.g., logistics vs legal risk, signals vs baselines, etc.).
3) The "Consider:" list must contain EXACTLY 10 items, comma-separated, short, domain-relevant, no trailing period.
4) Each item MUST start exactly with "You an expert " and include the substrings "forecaster specialized in ", "Consider: ", and the trailing clause exactly as written above.
5) Output ONLY the Python list literal with DOUBLE-QUOTED strings. No prose, no backticks, no numbering, no extra lines before/after.

Now produce the list.
"""
        )
        persona_llm = self.get_llm("persona_generator", "llm") or self.get_llm(
            "default", "llm"
        )
        raw: str | None = None
        if persona_llm and hasattr(persona_llm, "invoke"):
            try:
                raw = await persona_llm.invoke(prompt)  # type: ignore[union-attr]
            except Exception as exc:
                logger.warning(
                    "Persona generation failed with error: %s. Falling back to static personas.",
                    exc,
                )
        headers: list[str] = []
        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list) and all(
                    isinstance(item, str) for item in parsed
                ):
                    headers = [item.strip() for item in parsed]
            except Exception:
                pass

            if not headers:
                matches = re.findall(
                    r'"(You an expert .*?)"', raw, flags=re.DOTALL
                )
                if matches:
                    headers = [m.replace("\n", " ").strip() for m in matches]

        if not headers:
            headers = [
                header.format(output_limit=output_limit)
                for header in FALLBACK_PERSONA_HEADERS
            ][:n]

        if len(headers) < n:
            headers = headers + headers[: max(0, n - len(headers))]
        if len(headers) > n:
            headers = headers[:n]

        personas: list[tuple[str, str]] = []
        for idx, header in enumerate(headers):
            match = re.search(
                r"You an expert\\s+(.*?)\\s+forecaster\\s+specialized\\s+in",
                header,
                flags=re.IGNORECASE,
            )
            domain = match.group(1) if match else "Forecaster"
            domain_slug = re.sub(r"[^A-Za-z0-9]+", "-", domain).strip("-")
            name = f"{domain_slug or 'Persona'}-{idx + 1}"
            personas.append((name, header))

        self._personas_by_question[key] = personas
        self._persona_cursor_by_question[key] = 0

    async def _get_persona(self, question: MetaculusQuestion) -> tuple[str, str]:
        key = self._get_question_key(question)
        await self._ensure_personas_for_question(question)
        personas = self._personas_by_question[key]
        idx = self._persona_cursor_by_question.get(key, 0) % max(1, len(personas))
        self._persona_cursor_by_question[key] = (idx + 1) % max(1, len(personas))
        return personas[idx]

    def _get_openrouter_client(self):
        if self._openrouter_client is not None:
            return self._openrouter_client

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in environment")

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import guarded
            raise RuntimeError(
                "openai package is required for OpenRouter usage"
            ) from exc

        self._openrouter_client = OpenAI(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
        return self._openrouter_client

    def _store_structured_research(
        self, question: MetaculusQuestion, research: str
    ) -> None:
        if not research:
            return
        key = self._get_question_key(question)
        self._structured_research_cache[key] = research

    async def _maybe_generate_structured_research(
        self,
        question: MetaculusQuestion,
        existing_research: str,
        model_name: str | None = None,
    ) -> str:
        cached = self._structured_research_cache.get(self._get_question_key(question))
        if cached:
            return cached

        try:
            client = self._get_openrouter_client()
        except RuntimeError as exc:
            if not self._warned_openrouter_failure:
                logger.warning(
                    "Unable to generate structured research without OpenRouter client: %s",
                    exc,
                )
                self._warned_openrouter_failure = True
            return existing_research

        chosen_model = model_name or "perplexity/sonar-deep-research"
        prompt = build_structured_research_prompt(question.question_text)

        def _invoke():
            return client.chat.completions.create(
                model=chosen_model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"usage": {"include": True}},
            )

        try:
            completion = await asyncio.to_thread(_invoke)
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("Structured research request failed: %s", exc)
            return existing_research

        content = completion.choices[0].message.content if completion.choices else ""
        if content:
            self._store_structured_research(question, content)
            return content
        return existing_research

    async def _run_perplexity_jsonlines(
        self, question: MetaculusQuestion, model_name: str | None
    ) -> str:
        client = self._get_openrouter_client()
        actual_model = model_name or "perplexity/sonar-pro"
        question_text = question.question_text
        px_prompt = f"""
You are a research assistant. Find all materially relevant news for the QUESTION below, not just recent items—include seminal or high-impact older coverage if it adds context.

REQUIREMENTS
- Search broadly (multi-hop) and expand key terms, synonyms, and entities.
- Cover the full timeline: earliest notable item → most recent updates.
- De-duplicate near-duplicates; keep the best source per event.
- Prefer reputable outlets; avoid paywalled summaries without original reporting.
- Normalize dates to ISO 8601 (YYYY-MM-DD).
- OUTPUT ONLY the following fields in JSON Lines (one JSON object per line), nothing else:
  {{"headline": "...", "brief_content": "...", "date": "YYYY-MM-DD"}}

QUESTION: {question_text}
"""

        def _invoke():
            completion = client.chat.completions.create(
                model=actual_model,
                messages=[{"role": "user", "content": px_prompt}],
            )
            return completion.choices[0].message.content if completion.choices else ""

        try:
            response = await asyncio.to_thread(_invoke)
        except Exception as exc:  # pragma: no cover - network failure path
            raise RuntimeError(
                f"Perplexity research failed for model {actual_model}: {exc}"
            ) from exc

        return response or ""

    def _should_use_ensemble(self) -> bool:
        return bool(self.ensemble_model_groups)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            researcher = self.get_llm("researcher")
            research = ""
            structured_model_hint: str | None = None

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
                You do not produce forecasts yourself.

                Question:
                {question.question_text}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            try:
                if isinstance(researcher, GeneralLlm):
                    research = await researcher.invoke(prompt)
                elif researcher == "asknews/news-summaries":
                    research = await AskNewsSearcher().get_formatted_news_async(
                        question.question_text
                    )
                elif researcher == "asknews/deep-research/medium-depth":
                    research = await AskNewsSearcher().get_formatted_deep_research(
                        question.question_text,
                        sources=["asknews", "google"],
                        search_depth=2,
                        max_depth=4,
                    )
                elif researcher == "asknews/deep-research/high-depth":
                    research = await AskNewsSearcher().get_formatted_deep_research(
                        question.question_text,
                        sources=["asknews", "google"],
                        search_depth=4,
                        max_depth=6,
                    )
                elif isinstance(researcher, str) and researcher.startswith(
                    "smart-searcher"
                ):
                    model_name = researcher.removeprefix("smart-searcher/")
                    searcher = SmartSearcher(
                        model=model_name,
                        temperature=0,
                        num_searches_to_run=2,
                        num_sites_per_search=10,
                        use_advanced_filters=False,
                    )
                    research = await searcher.invoke(prompt)
                elif isinstance(researcher, str) and researcher.startswith(
                    "perplexity/sonar-deep-research"
                ):
                    structured_model_hint = researcher
                    research = await self._maybe_generate_structured_research(
                        question, existing_research="", model_name=structured_model_hint
                    )
                elif researcher == "perplexity":
                    research = await self._run_perplexity_jsonlines(question, None)
                elif isinstance(researcher, str) and researcher.startswith("perplexity/"):
                    research = await self._run_perplexity_jsonlines(
                        question, researcher
                    )
                elif not researcher or researcher == "None":
                    research = ""
                else:
                    research = await self.get_llm("researcher", "llm").invoke(prompt)
            except Exception as exc:
                logger.warning("Research pipeline failed (%s); falling back to empty string.", exc)
                research = ""

            if structured_model_hint:
                self._store_structured_research(question, research)
            elif self._should_use_ensemble():
                research = await self._maybe_generate_structured_research(
                    question, research
                )

            logger.info("Found Research for URL %s:\n%s", question.page_url, research)
            return research

    async def _run_binary_ensemble(
        self, question: BinaryQuestion, context: str
    ) -> list[dict]:
        if not (self.ensemble_model_groups and context):
            return []
        try:
            client = self._get_openrouter_client()
        except RuntimeError as exc:
            logger.warning("Disabling ensemble due to missing client: %s", exc)
            return []
        return await asyncio.to_thread(
            run_openrouter_ensemble,
            client,
            self.ensemble_model_groups,
            question.question_text,
            context,
        )

    @staticmethod
    def _limit_reasoning(text: str, limit: int = 480) -> str:
        snippet = text.strip()
        if not snippet:
            return ""
        if len(snippet) <= limit:
            return snippet
        return snippet[: limit - 1].rstrip() + "…"

    @staticmethod
    def _as_percent(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{round(value * 100, 2)}%"

    def _format_ensemble_reasoning(
        self,
        persona_name: str,
        persona_header: str,
        results: list[dict],
        aggregated_probability: float,
    ) -> str:
        lines: list[str] = [persona_header, "", f"Persona: {persona_name}"]
        if results:
            lines.append(
                "Ensemble members (model → probability, reasoning excerpt):"
            )
            for item in results:
                prob_text = self._as_percent(item.get("probability"))
                lines.append(f"{item.get('model')}: {prob_text}")
                excerpt = self._limit_reasoning(item.get("reasoning", ""))
                if excerpt:
                    lines.append(excerpt)
        lines.append(f"Probability: {self._as_percent(aggregated_probability)}")
        return "\n\n".join(lines)

    def _record_ensemble_results(
        self, question: BinaryQuestion, results: list[dict], context: str
    ) -> None:
        key = self._get_question_key(question)
        self._ensemble_results_by_question[key] = [
            {
                "model": item.get("model"),
                "probability": item.get("probability"),
                "reasoning": item.get("reasoning"),
                "context": self._limit_reasoning(context, limit=2000),
            }
            for item in results
        ]

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        persona_name, persona_header = await self._get_persona(question)
        logger.info(
            "Persona (binary) for %s: %s | %s",
            question.page_url,
            persona_name,
            persona_header,
        )

        if self._should_use_ensemble():
            structured_context = await self._maybe_generate_structured_research(
                question, research
            )
            results = await self._run_binary_ensemble(question, structured_context)
            aggregated_probability = _aggregate_probabilities(results)
            if aggregated_probability is None:
                aggregated_probability = 0.5
            aggregated_probability = max(0.01, min(0.99, aggregated_probability))
            reasoning = self._format_ensemble_reasoning(
                persona_name, persona_header, results, aggregated_probability
            )
            self._record_ensemble_results(question, results, structured_context)
            logger.info(
                "Forecasted URL %s with ensemble prediction: %s",
                question.page_url,
                aggregated_probability,
            )
            return ReasonedPrediction(
                prediction_value=aggregated_probability, reasoning=reasoning
            )

        prompt = clean_indents(
            f"""
            {persona_header}

            Begin your answer with exactly this line:
            Persona: {persona_name}

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            "Forecasted URL %s with prediction: %s",
            question.page_url,
            decimal_pred,
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        persona_name, persona_header = await self._get_persona(question)
        logger.info(
            "Persona (multiple choice) for %s: %s | %s",
            question.page_url,
            persona_name,
            persona_header,
        )
        prompt = clean_indents(
            f"""
            {persona_header}

            Begin your answer with exactly this line:
            Persona: {persona_name}

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

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
            "Forecasted URL %s with prediction: %s",
            question.page_url,
            predicted_option_list,
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
        persona_name, persona_header = await self._get_persona(question)
        logger.info(
            "Persona (numeric) for %s: %s | %s",
            question.page_url,
            persona_name,
            persona_header,
        )
        prompt = clean_indents(
            f"""
            {persona_header}

            Begin your answer with exactly this line:
            Persona: {persona_name}

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
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
            "Forecasted URL %s with prediction: %s",
            question.page_url,
            prediction.declared_percentiles,
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
            upper_bound_message = (
                f"The question creator thinks the number is likely not higher than {upper_bound_number}."
            )
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {upper_bound_number}."
            )

        if question.open_lower_bound:
            lower_bound_message = (
                f"The question creator thinks the number is likely not lower than {lower_bound_number}."
            )
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
        predictions_per_research_report=int(
            os.getenv("FORECAST_ENSEMBLE_SIZE", "3")
        ),
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o-mini",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openrouter/openai/gpt-4o-mini",
            "researcher": "perplexity/sonar-deep-research",
            "parser": "openrouter/openai/gpt-4o-mini",
            "persona_generator": "openrouter/openai/gpt-5",
        },
        output_limit=int(os.getenv("FORECAST_OUTPUT_LIMIT", "250")),
        prompt_variants=None,
        ensemble_model_groups=DEFAULT_ENSEMBLE_MODEL_GROUPS,
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
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
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
