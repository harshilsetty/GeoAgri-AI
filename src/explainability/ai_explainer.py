import json
import os
from typing import Any, Dict

import requests


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _soil_to_text(soil: Any) -> str:
    if isinstance(soil, str):
        return soil
    mapping = {1: "sandy", 2: "loam", 3: "clay"}
    try:
        return mapping.get(int(soil), "unknown")
    except (TypeError, ValueError):
        return "unknown"


def _normalize_key(raw_key: str | None) -> str:
    key = (raw_key or "").strip().strip('"').strip("'")
    if key.lower().startswith("bearer "):
        key = key.split(None, 1)[1].strip()

    # Some pasted keys include accidental whitespace/newlines.
    if key.startswith("gsk_"):
        key = "".join(key.split())

    return key


def build_prompt(features: Dict[str, Any], erosion_prob: float) -> str:
    return f"""
You are an environmental expert AI.

Analyze the terrain data below:

Slope: {_safe_float(features.get('slope')):.2f}
Vegetation: {_safe_float(features.get('vegetation')):.2f}
Rainfall: {_safe_float(features.get('rainfall')):.2f}
Soil Type: {_soil_to_text(features.get('soil'))}
Boulders: {_safe_float(features.get('boulders')):.2f}
Ruins: {_safe_float(features.get('ruins')):.2f}
Structures: {_safe_float(features.get('structures')):.2f}

Erosion Probability: {_safe_float(erosion_prob) * 100:.2f}%

Explain clearly:
- Why erosion risk is high or low
- What factors contributed most
- Keep it short (3-4 lines)
""".strip()


def generate_ai_explanation(features: Dict[str, Any], erosion_prob: float) -> str:
    slope = _safe_float(features.get("slope"))
    vegetation = _safe_float(features.get("vegetation"))
    rainfall = _safe_float(features.get("rainfall"))
    soil = _soil_to_text(features.get("soil"))

    reasons = []
    if rainfall > 100:
        reasons.append("high rainfall")
    if slope > 20:
        reasons.append("steep slope")
    if vegetation < 0.3:
        reasons.append("low vegetation cover")
    if soil == "sandy":
        reasons.append("erosion-prone sandy soil")

    risk = "high" if erosion_prob >= 0.7 else "moderate" if erosion_prob >= 0.4 else "low"
    reason_text = ", ".join(reasons[:3]) if reasons else "balanced terrain indicators"

    return (
        f"Estimated erosion risk is {risk} ({erosion_prob * 100:.1f}%). "
        f"The strongest contributors are {reason_text}. "
        "Use vegetation restoration, runoff control, and periodic monitoring to reduce future erosion."
    )


def generate_groq_explanation(features: Dict[str, Any], erosion_prob: float, api_key: str | None = None) -> str:
    text, _ = generate_groq_explanation_with_mode(features, erosion_prob, api_key=api_key)
    return text


def generate_groq_explanation_with_status(
    features: Dict[str, Any],
    erosion_prob: float,
    api_key: str | None = None,
) -> tuple[str, str, str | None]:
    prompt = build_prompt(features, erosion_prob)

    return generate_groq_from_prompt_with_status(
        prompt,
        fallback_text=generate_ai_explanation(features, erosion_prob),
        api_key=api_key,
    )


def generate_groq_explanation_with_mode(
    features: Dict[str, Any],
    erosion_prob: float,
    api_key: str | None = None,
) -> tuple[str, str]:
    text, mode, _ = generate_groq_explanation_with_status(features, erosion_prob, api_key=api_key)
    return text, mode


def generate_groq_from_prompt(prompt: str, fallback_text: str, api_key: str | None = None) -> str:
    text, _ = generate_groq_from_prompt_with_mode(prompt, fallback_text, api_key=api_key)
    return text


def generate_groq_from_prompt_with_status(
    prompt: str,
    fallback_text: str,
    api_key: str | None = None,
) -> tuple[str, str, str | None]:
    key = _normalize_key(api_key or os.getenv("GROQ_API_KEY"))
    if not key:
        return fallback_text, "local", "Missing GROQ API key. Set GROQ_API_KEY or provide a custom key."

    body = {
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise environmental risk analyst.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "geo-ai-ui/1.0",
            },
            json=body,
            timeout=25,
        )

        if response.status_code >= 400:
            detail_message = ""
            try:
                parsed = response.json() if response.text else {}
                detail_message = str((parsed.get("error") or {}).get("message") or "").strip()
            except Exception:
                detail_message = ""

            if response.status_code in (401, 403):
                reason = f"GROQ auth failed ({response.status_code})."
                if detail_message:
                    reason = f"{reason} {detail_message}"
                return fallback_text, "local", reason

            if response.status_code == 429:
                reason = "GROQ rate limit hit (429). Try again in a few seconds."
                if detail_message:
                    reason = f"{reason} {detail_message}"
                return fallback_text, "local", reason

            reason = f"GROQ request failed ({response.status_code})."
            if detail_message:
                reason = f"{reason} {detail_message}"
            return fallback_text, "local", reason

        payload = response.json()

        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            text = str(message.get("content") or "").strip()
            if text:
                return text, "groq", None

        return fallback_text, "local", "GROQ returned an empty completion."
    except requests.Timeout:
        return fallback_text, "local", "GROQ request timed out after 25 seconds."
    except Exception as e:
        return fallback_text, "local", f"GROQ exception: {type(e).__name__}: {str(e)}"


def generate_groq_from_prompt_with_mode(
    prompt: str,
    fallback_text: str,
    api_key: str | None = None,
) -> tuple[str, str]:
    text, mode, _ = generate_groq_from_prompt_with_status(prompt, fallback_text, api_key=api_key)
    return text, mode
