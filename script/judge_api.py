"""
LLM-as-a-Judge for EBA/ESMA Regulatory Q&A  (API version)
===========================================================
Calls the SwissAI serving API — no GPU memory required.

Pipeline:
1. Load questions from a data/splits/*_questions.jsonl file
2. Run Answerer 1 (Apertus-70B)       -> answer_1
3. Run Answerer 2 (Llama-3.3-70B)     -> answer_2
4. Run 3 judges (pointwise scoring) on each answer

All scores are stored as-is (per answerer, per judge); aggregation
(e.g. averaging across judges) happens downstream.

Usage:
    python judge_api.py --split val --n_samples 5

    python judge_api.py --split test \
        --answerers swiss-ai/Apertus-70B-Instruct-2509 meta-llama/Llama-3.3-70B-Instruct \
        --judges google/gemma-4-31B-it Qwen/Qwen3.5-27B zai-org/GLM-4.7-Flash
"""

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Model selection ───────────────────────────────────────────────────────────

ANSWERER_MODELS = ["swiss-ai/Apertus-70B-Instruct-2509", "meta-llama/Llama-3.3-70B-Instruct"]
JUDGE_MODELS    = ["google/gemma-4-31B-it", "Qwen/Qwen3.5-27B", "zai-org/GLM-4.7-Flash"]

# GLM and Qwen are reasoning models that burn tokens on hidden chain-of-thought
# before emitting visible content, regardless of the thinking-disable flags in
# _thinking_kwargs (which this endpoint doesn't appear to honor). Give them a
# much larger budget so they don't get cut off before producing the JSON.
JUDGE_MAX_TOKENS = {
    "Qwen/Qwen3.5-27B": 6144,
    "zai-org/GLM-4.7-Flash": 8192,
}
DEFAULT_JUDGE_MAX_TOKENS = 3072

SPLITS_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"


# ── Client ────────────────────────────────────────────────────────────────────

_client: OpenAI = OpenAI(
    api_key=os.environ.get("CSCS_SERVING_API"),
    base_url="https://api.swissai.svc.cscs.ch/v1",
)


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Drop messages with None/empty content that some endpoints reject."""
    return [m for m in messages if m.get("content")]


def run_multiturn_inference(model_name: str, messages: list, max_new_tokens: int = 512) -> str:
    """Call the API with a full conversation history and return the response text."""
    response = _client.chat.completions.create(
        model=model_name,
        messages=_sanitize_messages(messages),
        max_tokens=max_new_tokens,
    )
    return response.choices[0].message.content.strip()


_REASONING_MODEL_MARKERS = ("qwen", "glm")


def _is_reasoning_model(model_name: str) -> bool:
    name = model_name.lower()
    return any(marker in name for marker in _REASONING_MODEL_MARKERS)


def _thinking_kwargs(model_name: str) -> dict:
    """Best-effort flags to disable extended/chain-of-thought reasoning.

    Different serving backends expose this differently (vLLM's
    chat_template_kwargs for Qwen, zai's "thinking" field for GLM, etc).
    Unsupported fields are typically ignored by the server rather than
    rejected, but if the endpoint errors on an unknown field, drop the
    relevant branch. These have proven unreliable on the SwissAI endpoint
    (GLM in particular ignores them), so they're combined with the
    "/no_think" prompt-suffix trick in generate() as a second line of
    defense.
    """
    name = model_name.lower()
    if "qwen" in name:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if "glm" in name:
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def _call_model(model_name: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
    # "/no_think" is parsed by several vLLM-served Qwen3/GLM chat templates
    # to suppress the <think> block at the template level, independent of
    # whatever extra_body flags the server does or doesn't wire up.
    if _is_reasoning_model(model_name):
        prompt = f"{prompt}\n\n/no_think"
    messages = [{"role": "user", "content": prompt}]
    response = _client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        **_thinking_kwargs(model_name),
    )
    msg = response.choices[0].message
    # Some reasoning models (Kimi, DeepSeek-R1) return None for content
    # and put the actual reply in reasoning_content or a custom field.
    text = msg.content or getattr(msg, "reasoning_content", None) or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    finish_reason = response.choices[0].finish_reason
    if not text:
        raise ValueError(
            f"Empty response from {model_name} (finish_reason={finish_reason}): {response}"
        )
    return text


def generate(model_name: str, prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    try:
        return _call_model(model_name, prompt, max_new_tokens, temperature)
    except ValueError:
        if not _is_reasoning_model(model_name):
            raise
        # Reasoning models occasionally burn the whole budget on hidden
        # chain-of-thought with no visible output; retry once with a much
        # larger budget before giving up.
        return _call_model(model_name, prompt, max_new_tokens * 2, temperature)


# ── Prompts ───────────────────────────────────────────────────────────────────

ANSWERER_PROMPT = """\
You are an expert in EU financial regulation, with deep knowledge of \
EBA and ESMA guidelines, technical standards, and related directives \
and regulations.

You will receive a regulatory question along with structured context:
- LEGAL ACT: the specific instrument the question concerns
- TOPIC: the regulatory area
- SUBJECT MATTER: the specific provision, field, or requirement
- BACKGROUND: the context or inconsistency motivating the question

## Requirements

1. **ACCURACY**: Base your answer on the actual content of the legal \
act identified above and any directly relevant secondary instruments \
(RTS, ITS, guidelines). Address the specific subject matter and \
background; do not answer a more general version of the question.

2. **CITATIONS**: Every substantive claim must be supported by a \
specific citation in the form [Source, Article/Paragraph/Field], e.g., \
[Regulation (EU) 2022/2554, Art. 28(3)] or [EBA/ITS/2023/01, field \
B_05.01.0020]. The primary citation should point to the legal act \
named in the context. General references such as "DORA" or "the ITS" \
are not sufficient. Use square brackets consistently.

3. **ABSTENTION**: If you cannot identify a specific provision for a \
claim, write "[no specific provision identified]" immediately after \
the claim, rather than fabricating a citation. Partial answers with \
honest gaps are preferred over complete answers with fabricated \
references.

4. **FORMAT**: 100-400 words of prose, matching the style of official \
EBA/ESMA Q&A responses. Inline citations after each substantive \
claim. Do not pad.

5. **CONCISENESS**: State each point once. Do not restate the question, \
do not write a closing "In summary" / "In conclusion" paragraph that \
repeats what you already said, and do not hedge with filler like "it \
is worth noting" or "however, it should be noted". If you do not know \
something, say so plainly in one sentence rather than reasoning around \
it at length.

## Context

LEGAL ACT: {legal_act}
TOPIC: {topic}
SUBJECT MATTER: {subject_matter}
BACKGROUND: {background}

## Question

{question}

Answer :
"""

JUDGE_PROMPT = """\
You are an expert judge evaluating answers to EU financial regulation \
questions from EBA and ESMA sources. You will score a candidate answer \
against the official answer on four dimensions.

## Question
{question}

## Topic
{topic}

## Subject Matter
{subject_matter}

## Official Answer (Ground Truth)
{ground_truth}

## Candidate Answer
{candidate}

## Scoring Dimensions

**Accuracy** (factual correctness relative to the official answer):
- 5: All factual claims match the official answer.
- 4: Minor inaccuracies that do not change the substantive conclusion.
- 3: Partially correct; one substantive claim is wrong or unsupported.
- 2: Multiple substantive errors; conclusion is partly incorrect.
- 1: Conclusion contradicts the official answer or is fabricated.

**Completeness** (coverage of key points in the official answer):
- 5: Covers all key points present in the official answer.
- 4: Covers all key points but omits a minor detail.
- 3: Covers the main point but misses one secondary point.
- 2: Misses multiple key points; partial coverage.
- 1: Misses the main point entirely.

**Topic Coherence** (alignment with the specified topic and subject matter):
- 5: Fully addresses the specified topic and subject matter without \
drifting into adjacent regulatory areas.
- 4: Stays on topic but includes minor tangential content, OR addresses \
a broader scope that still fully covers the topic.
- 3: Partially on topic; some content addresses a different but related \
regulatory area.
- 2: Primarily addresses an adjacent topic; only partially relevant to \
the specified subject matter.
- 1: Off-topic or addresses a different regulatory area entirely.

**Citation Quality** (specificity and correctness of legal references):
- 5: All citations are specific (article/paragraph/field level) and \
correctly identify the supporting provision.
- 4: Citations are specific and mostly correct; one minor citation issue.
- 3: Citations are present but partially generic (e.g., "Article 5" \
without specifying the regulation), or one citation is incorrect.
- 2: Citations are mostly generic ("the ITS", "DORA") or several are \
incorrect.
- 1: Citations are missing or fabricated.

## Instructions

1. **Reason before scoring.** Think step-by-step about each dimension \
before assigning the score. Your reasoning should determine the score, \
not the reverse.

2. **Catch plausible-sounding hallucinations.** If the candidate \
contains specific factual claims (numbers, dates, article references, \
requirements, definitions) that are not supported by the official \
answer, treat those as accuracy violations even if the claims sound \
plausible.

3. **Score dimensions independently.** If two dimensions seem to \
conflict, score each on its own merits.

4. **Use the full 1-5 range.** Do not default to 3 when uncertain. \
A shorter candidate that still covers the key point is not a \
completeness penalty.

4b. **Score must match your own reasoning.** If your reasoning \
sentence for a dimension names a specific flaw (a missing point, a \
wrong citation, an unsupported claim, drift off-topic), the score for \
that dimension cannot be 5. Reserve 5 only when your reasoning states \
the candidate fully and correctly meets the criterion with no caveat. \
Do not default to 5 out of leniency.

5. **Citation specificity matters.** A correct citation to "Article 5 \
of Regulation (EU) 2019/2033" is stronger than "Article 5" alone, \
even when both are technically correct.

6. **Be concise.** Do your step-by-step thinking silently. Each \
reasoning field must be ONE short sentence (max ~25 words) stating the \
conclusion, not a transcript of your deliberation. Do not repeat the \
question, the candidate text, or the official answer back. Output the \
JSON object immediately after you have decided the scores — no \
preamble, no text before or after the JSON.

7. **No visible thinking.** Do not output a `<think>` block, chain-of-\
thought, or any reasoning outside the JSON's "reasoning" fields. Your \
entire response must be the JSON object and nothing else.

Respond in this exact JSON format only:
{{
  "reasoning": {{
    "accuracy": "<one sentence>",
    "completeness": "<one sentence>",
    "topic_coherence": "<one sentence>",
    "citation_quality": "<one sentence>"
  }},
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "topic_coherence": <1-5>,
  "citation_quality": <1-5>
}}"""


# ── Score parsing ─────────────────────────────────────────────────────────────

SCORE_KEYS = {"accuracy", "completeness", "topic_coherence", "citation_quality"}

def _find_balanced_objects(text: str) -> list[str]:
    """Return all top-level {...} substrings, respecting brace nesting."""
    blocks = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    blocks.append(text[start:i + 1])
    return blocks


def parse_scores(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    for block in reversed(_find_balanced_objects(text)):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if SCORE_KEYS.issubset(data.keys()) and all(
            isinstance(data[k], (int, float)) for k in SCORE_KEYS
        ):
            return data
    return {"accuracy": None, "completeness": None, "topic_coherence": None,
            "citation_quality": None, "reasoning": text[-300:]}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(split: str) -> list[dict]:
    path = SPLITS_DIR / f"{split}_questions.jsonl"
    with open(path, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    items = [x for x in items if x.get("question") and x.get("answer")]
    print(f"Loaded {len(items)} Q&As from {path}")
    return items


# ── Pipeline ──────────────────────────────────────────────────────────────────

def judge_answer(judge_model: str, item: dict, candidate: str) -> dict:
    """Pointwise-score a single candidate answer with a single judge."""
    try:
        judge_out = generate(
            judge_model,
            JUDGE_PROMPT.format(
                question=item["question"],
                topic=item.get("topic", ""),
                subject_matter=item.get("subject_matter", ""),
                ground_truth=item["answer"],
                candidate=candidate,
            ),
            max_new_tokens=JUDGE_MAX_TOKENS.get(judge_model, DEFAULT_JUDGE_MAX_TOKENS),
            temperature=0,
        )
        return parse_scores(judge_out)
    except Exception as e:
        print(f"    Judge {judge_model} error: {e} — recording null scores.")
        return {"accuracy": None, "completeness": None, "topic_coherence": None,
                "citation_quality": None, "reasoning": str(e)}


def run(qa_items: list[dict], answerer_models: list[str], judge_models: list[str],
        output_path: str, done_ids: set) -> list[dict]:
    results = []
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    remaining = [item for item in qa_items if item["question_id"] not in done_ids]
    skipped = len(qa_items) - len(remaining)
    if skipped:
        print(f"Resuming: skipping {skipped} already-processed items.")

    for i, item in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {item['question_id']}")

        answers = {}
        for answerer_model in answerer_models:
            try:
                candidate = generate(
                    answerer_model,
                    ANSWERER_PROMPT.format(
                        legal_act=item.get("legal_act", ""),
                        topic=item.get("topic", ""),
                        subject_matter=item.get("subject_matter", ""),
                        background=item.get("background", ""),
                        question=item["question"],
                    ),
                )
            except Exception as e:
                print(f"  Answerer {answerer_model} error: {e} — skipping answerer.")
                continue
            print(f"  [{answerer_model}] {candidate[:120]}...")

            judge_scores = {}
            for judge_model in judge_models:
                scores = judge_answer(judge_model, item, candidate)
                judge_scores[judge_model] = scores
                print(f"    [{judge_model}] accuracy={scores['accuracy']} "
                      f"completeness={scores['completeness']} "
                      f"topic_coherence={scores['topic_coherence']} "
                      f"citation_quality={scores['citation_quality']}")

            answers[answerer_model] = {"text": candidate, "judge_scores": judge_scores}

        result = {
            "question_id": item["question_id"],
            "regulator": item.get("regulator", ""),
            "question": item["question"],
            "ground_truth": item["answer"],
            "answers": answers,
            "meta": {
                "answerer_models": answerer_models,
                "judge_models": judge_models,
                "status": item.get("status", ""),
                "legal_act": item.get("legal_act", ""),
                "topic": item.get("topic", ""),
                "subject_matter": item.get("subject_matter", ""),
                "url": item.get("url", ""),
            },
        }
        results.append(result)

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_split(split: str, answerers: list[str], judges: list[str], n_samples: int) -> list[dict]:
    all_items = load_split(split)
    samples = all_items[:n_samples]

    out_path = Path(f"output/judge_results_{split}_api.jsonl")

    done_ids: set = set()
    prior_results: list[dict] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            prior_results = [json.loads(l) for l in f if l.strip()]
        done_ids = {r["question_id"] for r in prior_results}
        print(f"Loaded {len(prior_results)} checkpointed results from {out_path}")

    new_results = run(samples, answerers, judges, str(out_path), done_ids)
    all_results = prior_results + new_results

    print(f"Total results: {len(all_results)} ({len(new_results)} new) -> {out_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="Which data/splits/<split>_questions.jsonl file to process")
    parser.add_argument("--answerers", nargs="+", default=ANSWERER_MODELS,
                        help="Answerer model names served at the SwissAI endpoint")
    parser.add_argument("--judges", nargs="+", default=JUDGE_MODELS,
                        help="Judge model names")
    parser.add_argument("--n_samples", type=int, default=9999,
                        help="Max number of questions to process from the split")
    args = parser.parse_args()

    print(f"Split     : {args.split}")
    print(f"Answerers : {args.answerers}")
    print(f"Judges    : {args.judges}")
    print(f"Endpoint  : {_client.base_url}")

    run_split(args.split, args.answerers, args.judges, args.n_samples)


if __name__ == "__main__":
    main()
