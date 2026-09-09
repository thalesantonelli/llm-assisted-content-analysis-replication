"""Claim Detection through Semantic Clustering.

Public implementation of the computational stages described in the article:
claim extraction, sequential analytical filtering, weak semantic normalization,
and multi-resolution semantic clustering.

Corpus construction, taxonomic classification, and interpretive validation are
researcher-led stages and are not automated in this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    from google import genai
except Exception as exc:
    raise ImportError(
        "google-genai is required. Install the dependencies listed for the project."
    ) from exc

try:
    import hdbscan
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
except Exception as exc:
    raise ImportError(
        "scikit-learn and hdbscan are required for semantic clustering."
    ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd() / "claim_pipeline")).resolve()

PIPELINE_DIRS = {
    "inputs": PROJECT_ROOT / "00_inputs",
    "claims": PROJECT_ROOT / "01_claim_extraction",
    "filters": PROJECT_ROOT / "02_analytical_filtering",
    "normalization": PROJECT_ROOT / "03_weak_semantic_normalization",
    "clustering": PROJECT_ROOT / "04_semantic_clustering",
    "logs": PROJECT_ROOT / "99_logs",
}

for directory in PIPELINE_DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

RAW_INPUT_PATH = PIPELINE_DIRS["inputs"] / "speeches_input.xlsx"
CLAIMS_OUTPUT_PATH = PIPELINE_DIRS["claims"] / "claims_extracted.xlsx"
FILTERED_OUTPUT_PATH = PIPELINE_DIRS["filters"] / "claims_filtered.xlsx"
NORMALIZED_OUTPUT_PATH = PIPELINE_DIRS["normalization"] / "claims_normalized.xlsx"
CLUSTER_OUTPUT_PATH = PIPELINE_DIRS["clustering"] / "claims_clustered.xlsx"

ID_COL = "ID"
TEXT_COL = "Discurso Completo"
CLAIM_COL = "claim"
NORMALIZED_COL = "normalized_claim"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_TEMPERATURE = 0.0

MAX_WORKERS_EXTRACTION = 5
MAX_WORKERS_FILTERING = 5
MAX_WORKERS_NORMALIZATION = 8

SVD_COMPONENTS = 100
RANDOM_STATE = 42

# Multi-resolution HDBSCAN configurations used in the final analysis.
CLUSTER_CONFIGS = {
    "macro": {
        "min_cluster_size": 40,
        "min_samples": 15,
        "cluster_selection_method": "eom",
    },
    "meso": {
        "min_cluster_size": 25,
        "min_samples": 10,
        "cluster_selection_method": "eom",
    },
    "micro": {
        "min_cluster_size": 12,
        "min_samples": 6,
        "cluster_selection_method": "leaf",
    },
}


# -----------------------------------------------------------------------------
# Shared utilities
# -----------------------------------------------------------------------------


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not defined in the environment.")
    return genai.Client(api_key=api_key)


def load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_excel(path)
    print(f"Loaded {path} | rows={len(df):,} | columns={len(df.columns)}")
    return df


def save_excel(df: pd.DataFrame, path: Path, sheet_name: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, index=False, sheet_name=sheet_name)
    print(f"Saved {path} | rows={len(df):,}")
    return path


def resolve_column(
    df: pd.DataFrame,
    preferred: str,
    alternatives: Optional[List[str]] = None,
) -> str:
    for column in [preferred] + (alternatives or []):
        if column in df.columns:
            return column
    raise KeyError(f"None of the expected columns were found: {[preferred] + (alternatives or [])}")


def clean_one_line(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_json_block(text: str) -> str:
    if not text:
        return ""

    fenced = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1].strip()

    return text.strip()


def light_repair_json(text: str) -> str:
    if not text:
        return text
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("\ufeff", "").replace("\u00a0", " ")
    text = re.sub(r",\s*(\})", r"\1", text)
    text = re.sub(r",\s*(\])", r"\1", text)
    return text.strip()


def parse_json_lenient(
    text: str,
    normalizer: Callable[[Any, str], Tuple[Optional[Dict[str, Any]], bool, Optional[str]]],
):
    if text is None:
        return None, True, "empty_response", ""

    raw = text.strip()
    extracted = extract_json_block(raw)
    attempts = [raw, extracted, light_repair_json(extracted), light_repair_json(raw)]
    attempts = [candidate for i, candidate in enumerate(attempts) if candidate and candidate not in attempts[:i]]

    last_error = None
    for candidate in attempts:
        try:
            obj = json.loads(candidate)
            normalized, error, message = normalizer(obj, raw)
            return normalized, error, message, candidate
        except Exception as exc:
            last_error = exc

    return None, True, f"json_decode_failed:{last_error}", light_repair_json(raw)


def gemini_generate(prompt: str, response_json: bool = False) -> str:
    client = get_client()
    config: Dict[str, Any] = {"temperature": GEMINI_TEMPERATURE}
    if response_json:
        config["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )

    raw = getattr(response, "text", None)
    if raw:
        return raw.strip()

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(part, "text", "") for part in parts).strip()


def run_parallel(
    items: List[Any],
    worker_fn: Callable[[Any], Dict[str, Any]],
    max_workers: int,
    log_every: int = 25,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    errors = 0
    started = time.time()
    total = len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_fn, item) for item in items]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {"ok": False, "parse_error": True, "error_message": str(exc)}

            results.append(result)
            if result.get("parse_error") or result.get("ok") is False:
                errors += 1

            done = len(results)
            if done % log_every == 0 or done == total:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0.0
                print(
                    f"Processed {done:,}/{total:,} | errors={errors:,} | "
                    f"rate={rate * 60:.1f} items/min"
                )

    return results


# -----------------------------------------------------------------------------
# 1. LLM-assisted claim extraction
# -----------------------------------------------------------------------------

CLAIM_EXTRACTION_PROMPT = """
Task: extract all analytically relevant claims from the political text below.

A claim is a position-bearing unit of substantive justification: a statement, premise,
diagnosis, evaluation, proposal, causal attribution, or judgment that sustains a political
position about what is legitimate, desirable, threatening, feasible, necessary, unjust, or
problematic. Claims may be explicit or implicit, but an implicit claim must be directly
supported by the text.

Extraction rules:
1. Read the complete text before extracting claims.
2. Extract distinct claims separately.
3. Preserve the substantive meaning of the source while expressing each claim as a concise,
   self-contained proposition.
4. Exclude ceremonial language, dates, isolated factual details, and purely descriptive
   statements unless they sustain a position.
5. Do not attribute ideas that are not supported by the text.
6. Do not collapse substantively different claims into a single item.

Return valid JSON only:
{{
  "claims": [
    "first claim",
    "second claim"
  ]
}}

If no claim is present, return an empty list.

Text:
----------------------------
{text}
----------------------------
""".strip()


def normalize_claim_extraction(
    obj: Any,
    raw_text: str = "",
) -> Tuple[Optional[Dict[str, List[str]]], bool, Optional[str]]:
    if not isinstance(obj, dict):
        return None, True, "json_not_object"

    claims = obj.get("claims")
    if isinstance(claims, str):
        claims = [claims.strip()] if claims.strip() else []

    if not isinstance(claims, list):
        return None, True, "invalid_claims_field"

    cleaned = []
    for item in claims:
        value = clean_one_line(item)
        if value and value.lower() not in {"none", "null"}:
            cleaned.append(value)

    return {"claims": cleaned}, False, None


def extract_claims_from_speech(speech_id: Any, speech_text: str) -> List[Dict[str, Any]]:
    prompt = CLAIM_EXTRACTION_PROMPT.format(text=str(speech_text or ""))
    raw = gemini_generate(prompt, response_json=True)
    parsed, error, message, _ = parse_json_lenient(raw, normalize_claim_extraction)

    if error or not parsed:
        return [{
            "speech_id": speech_id,
            "speech_text": speech_text,
            "claim_id": None,
            "claim": None,
            "parse_error": True,
            "error_message": message,
            "raw_response": raw,
        }]

    rows: List[Dict[str, Any]] = []
    for number, claim in enumerate(parsed["claims"], start=1):
        rows.append({
            "speech_id": speech_id,
            "speech_text": speech_text,
            "claim_id": f"{speech_id}_{number}",
            "claim": claim,
            "parse_error": False,
            "error_message": "",
            "raw_response": raw if number == 1 else "",
        })

    if not rows:
        rows.append({
            "speech_id": speech_id,
            "speech_text": speech_text,
            "claim_id": None,
            "claim": None,
            "parse_error": False,
            "error_message": "no_claims",
            "raw_response": raw,
        })

    return rows


def run_claim_extraction() -> pd.DataFrame:
    df = load_excel(RAW_INPUT_PATH)
    id_col = resolve_column(df, ID_COL, ["id", "speech_id", "proferimento_id", "codigo", "Código"])
    text_col = resolve_column(
        df,
        TEXT_COL,
        ["text", "Text", "speech_text", "texto", "Texto", "proferimento_texto", "discurso", "Discurso"],
    )

    items = [(row[id_col], row[text_col]) for _, row in df.iterrows()]

    def worker(item):
        speech_id, speech_text = item
        return extract_claims_from_speech(speech_id, speech_text)

    nested = run_parallel(items, worker, MAX_WORKERS_EXTRACTION)
    rows = [row for group in nested for row in group]
    claims_df = pd.DataFrame(rows)

    valid = claims_df[claims_df["claim"].notna()].copy()
    errors = claims_df[claims_df["parse_error"].eq(True)].copy()

    save_excel(valid, CLAIMS_OUTPUT_PATH, "claims")
    if not errors.empty:
        save_excel(errors, PIPELINE_DIRS["logs"] / "claim_extraction_errors.xlsx", "errors")

    return valid


# -----------------------------------------------------------------------------
# 2. Sequential analytical filtering
# -----------------------------------------------------------------------------

ANALYTICAL_FILTERS = [
    {
        "name": "land_territory_relevance",
        "definition": (
            "Retain claims substantively related to land and territorial politics, including "
            "agrarian reform, land ownership and access, Indigenous lands, quilombola territories, "
            "territorial rights, land conflicts, rural property, demarcation, occupation, or related "
            "institutional disputes."
        ),
    },
    {
        "name": "contestatory_orientation",
        "definition": (
            "Retain claims that contest, challenge, question, criticize, or delegitimize land and "
            "territorial demands, actors, policies, rights, or institutions by questioning their "
            "legitimacy, legality, rationality, desirability, feasibility, morality, or public value."
        ),
    },
    {
        "name": "climate_environment_repertoire",
        "definition": (
            "Retain claims in which climate or environmental considerations participate in the "
            "justificatory structure, including environmental protection, climate policy, "
            "sustainability, biodiversity, conservation, ecological regulation, natural resources, "
            "or related environmental repertoires. Mere incidental mention is insufficient."
        ),
    },
]

FILTER_PROMPT = """
Classify whether the claim satisfies the analytical criterion below.

Criterion:
{criterion}

Evaluate the substantive meaning of the claim rather than keyword presence alone. Return
"Yes" only when the criterion is part of the claim's meaning or justificatory function.

Claim:
----------------------------
{claim}
----------------------------

Return valid JSON only:
{{
  "classification": "Yes" or "No",
  "rationale": "brief justification"
}}
""".strip()


def normalize_filter_result(
    obj: Any,
    raw_text: str = "",
) -> Tuple[Optional[Dict[str, str]], bool, Optional[str]]:
    if not isinstance(obj, dict):
        return None, True, "json_not_object"

    classification = obj.get("classification")
    rationale = obj.get("rationale", "")
    if not isinstance(classification, str):
        return None, True, "invalid_classification_field"

    value = classification.strip().lower()
    aliases = {
        "yes": "Yes",
        "y": "Yes",
        "true": "Yes",
        "sim": "Yes",
        "no": "No",
        "n": "No",
        "false": "No",
        "não": "No",
        "nao": "No",
    }
    value = aliases.get(value, classification.strip())
    if value not in {"Yes", "No"}:
        return None, True, f"invalid_classification:{classification!r}"

    return {
        "classification": value,
        "rationale": clean_one_line(rationale),
    }, False, None


def classify_claim(claim: str, criterion: str) -> Dict[str, Any]:
    prompt = FILTER_PROMPT.format(criterion=criterion, claim=str(claim or ""))
    raw = gemini_generate(prompt, response_json=True)
    parsed, error, message, _ = parse_json_lenient(raw, normalize_filter_result)

    return {
        "classification": None if error or not parsed else parsed["classification"],
        "rationale": None if error or not parsed else parsed["rationale"],
        "parse_error": error,
        "error_message": message,
        "raw_response": raw,
    }


def run_analytical_filtering() -> pd.DataFrame:
    current = load_excel(CLAIMS_OUTPUT_PATH)
    claim_col = resolve_column(current, CLAIM_COL, ["Claim", "CLAIM", "claim_text", "texto_claim"])

    for stage_number, filter_spec in enumerate(ANALYTICAL_FILTERS, start=1):
        stage_name = filter_spec["name"]
        criterion = filter_spec["definition"]

        items = [(idx, current.at[idx, claim_col]) for idx in current.index]

        def worker(item):
            idx, claim = item
            result = classify_claim(claim, criterion)
            return {"idx": idx, **result}

        results = run_parallel(items, worker, MAX_WORKERS_FILTERING)
        results_df = pd.DataFrame(results).set_index("idx")

        prefix = f"filter_{stage_number}_{stage_name}"
        for source, suffix in [
            ("classification", "classification"),
            ("rationale", "rationale"),
            ("parse_error", "parse_error"),
            ("error_message", "error_message"),
            ("raw_response", "raw_response"),
        ]:
            current[f"{prefix}_{suffix}"] = results_df[source]

        stage_path = PIPELINE_DIRS["filters"] / f"stage_{stage_number}_{stage_name}.xlsx"
        save_excel(current, stage_path, "filtered_claims")

        retained = current[f"{prefix}_classification"].eq("Yes")
        current = current[retained].copy()
        print(f"Filter {stage_number}: {stage_name} | retained={len(current):,}")

    save_excel(current, FILTERED_OUTPUT_PATH, "analytical_corpus")
    return current


# -----------------------------------------------------------------------------
# 3. Weak semantic normalization
# -----------------------------------------------------------------------------

NORMALIZATION_PROMPT = """
Perform weak semantic normalization of the claim below.

Weak semantic normalization is a controlled reformulation intended to improve comparability
before clustering. Reduce surface phrasing, ellipsis, stylistic variation, and rhetorical
elaboration while preserving the original justificatory premise.

Requirements:
1. Preserve the substantive political meaning of the claim.
2. Do not add theoretical interpretation or information absent from the claim.
3. Remove contingent names, dates, locations, and rhetorical ornament when they are not
   necessary to preserve the premise.
4. Produce one concise, declarative, self-contained proposition.
5. Do not merge distinct premises or make the claim more general than the source supports.

Claim:
----------------------------
{claim}
----------------------------

Return valid JSON only:
{{
  "normalized_claim": "normalized proposition",
  "normalization_rationale": "brief justification"
}}
""".strip()


def normalize_normalization_result(
    obj: Any,
    raw_text: str = "",
) -> Tuple[Optional[Dict[str, str]], bool, Optional[str]]:
    if not isinstance(obj, dict):
        return None, True, "json_not_object"

    normalized_claim = obj.get("normalized_claim")
    rationale = obj.get("normalization_rationale")

    if not isinstance(normalized_claim, str) or not normalized_claim.strip():
        return None, True, "invalid_normalized_claim"
    if not isinstance(rationale, str) or not rationale.strip():
        return None, True, "invalid_normalization_rationale"

    return {
        "normalized_claim": clean_one_line(normalized_claim),
        "normalization_rationale": clean_one_line(rationale),
    }, False, None


def normalize_claim_text(claim: str, retries: int = 2) -> Tuple[bool, Dict[str, Any]]:
    if claim is None or not str(claim).strip():
        return False, {"error": "empty_claim"}

    prompt = NORMALIZATION_PROMPT.format(claim=str(claim).strip())
    last_error: Optional[Dict[str, Any]] = None

    for attempt in range(retries + 1):
        try:
            raw = gemini_generate(prompt, response_json=True)
            parsed, error, message, _ = parse_json_lenient(raw, normalize_normalization_result)
            if not error and parsed:
                return True, {**parsed, "raw_response": raw}
            last_error = {"error": message, "raw_response": raw[:500]}
        except Exception as exc:
            last_error = {"error": str(exc)}
        time.sleep(0.5 * (attempt + 1))

    return False, last_error or {"error": "unknown_error"}


def run_weak_semantic_normalization() -> pd.DataFrame:
    df = load_excel(FILTERED_OUTPUT_PATH)
    claim_col = resolve_column(df, CLAIM_COL, ["Claim", "CLAIM", "claim_text", "texto_claim"])

    items = [(idx, df.at[idx, claim_col]) for idx in df.index]

    def worker(item):
        idx, claim = item
        ok, output = normalize_claim_text(claim)
        return {"idx": idx, "ok": ok, "output": output}

    results = run_parallel(items, worker, MAX_WORKERS_NORMALIZATION)

    df[NORMALIZED_COL] = ""
    df["normalization_rationale"] = ""
    df["normalization_ok"] = False
    df["normalization_error"] = ""
    df["normalization_raw_response"] = ""

    for result in results:
        idx = result["idx"]
        output = result["output"]
        if result["ok"]:
            df.at[idx, NORMALIZED_COL] = output["normalized_claim"]
            df.at[idx, "normalization_rationale"] = output["normalization_rationale"]
            df.at[idx, "normalization_ok"] = True
            df.at[idx, "normalization_raw_response"] = output.get("raw_response", "")
        else:
            df.at[idx, "normalization_error"] = output.get("error", "unknown_error")

    save_excel(df, NORMALIZED_OUTPUT_PATH, "normalized_claims")
    return df


# -----------------------------------------------------------------------------
# 4. Semantic clustering
# -----------------------------------------------------------------------------


def run_semantic_clustering() -> pd.DataFrame:
    df = load_excel(NORMALIZED_OUTPUT_PATH)
    text_col = resolve_column(
        df,
        NORMALIZED_COL,
        ["claim_canonico", "canonical_claim", "canonico", "canônico"],
    )

    usable = df[df[text_col].fillna("").astype(str).str.strip().ne("")].copy()
    if usable.empty:
        raise RuntimeError("No normalized claims are available for clustering.")

    texts = usable[text_col].astype("string").fillna("").map(clean_one_line)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    x = vectorizer.fit_transform(texts)

    if x.shape[1] <= 1:
        z = x.toarray()
    else:
        n_components = min(SVD_COMPONENTS, x.shape[1] - 1, max(1, len(usable) - 1))
        z = TruncatedSVD(
            n_components=n_components,
            random_state=RANDOM_STATE,
        ).fit_transform(x)

    z = normalize(z)

    usable["cluster_method"] = "HDBSCAN"

    for level, params in CLUSTER_CONFIGS.items():
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric="euclidean",
            cluster_selection_method=params["cluster_selection_method"],
        )

        labels = clusterer.fit_predict(z)
        cluster_col = f"{level}_cluster_id"
        size_col = f"{level}_cluster_size"
        usable[cluster_col] = labels
        usable[size_col] = usable.groupby(cluster_col)[cluster_col].transform("size")

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(
            f"{level.upper()}: clusters={n_clusters:,} | "
            f"noise={n_noise:,}/{len(usable):,} ({n_noise / len(usable):.1%}) | "
            f"min_cluster_size={params['min_cluster_size']} | "
            f"min_samples={params['min_samples']} | "
            f"selection={params['cluster_selection_method']}"
        )

    save_excel(usable, CLUSTER_OUTPUT_PATH, "clusters")
    return usable


# -----------------------------------------------------------------------------
# Command-line entry point
# -----------------------------------------------------------------------------

STAGES = {
    "extract": run_claim_extraction,
    "filter": run_analytical_filtering,
    "normalize": run_weak_semantic_normalization,
    "cluster": run_semantic_clustering,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-assisted claim extraction, analytical filtering, normalization, and clustering."
    )
    parser.add_argument(
        "stage",
        choices=[*STAGES.keys(), "all"],
        help="Pipeline stage to execute.",
    )
    args = parser.parse_args()

    if args.stage == "all":
        for stage_name, function in STAGES.items():
            print(f"\n=== {stage_name.upper()} ===")
            function()
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Pipeline failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
