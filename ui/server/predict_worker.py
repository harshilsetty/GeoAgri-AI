import json
import sys
import traceback

from predict_bridge import generate_insights, predict, predict_point, recommend_crops


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_key(value):
    key = str(value or "").strip().strip('"').strip("'")
    if key.lower().startswith("bearer "):
        key = key.split(None, 1)[1].strip()
    return "".join(key.split())


def _handle_request(message):
    action = str(message.get("action") or "predict").strip().lower()

    if action in {"crop", "agriculture"}:
        return recommend_crops(
            lat=float(message.get("lat", message.get("latitude", 15.335))),
            lon=float(message.get("lon", message.get("longitude", 76.46))),
            season=str(message.get("season", "Kharif")),
            vegetation_ratio=float(message.get("vegetation", 0.35)),
            api_key=_normalize_key(message.get("apiKey") or ""),
            top_k=int(message.get("topK", message.get("top_k", 5))),
        )

    if action == "point":
        return predict_point(
            lat=float(message["lat"]),
            lon=float(message["lon"]),
            api_key=_normalize_key(message.get("apiKey") or ""),
            use_ai_insight=_to_bool(message.get("useAiInsight", False)),
            include_shap=_to_bool(message.get("includeShap", True)),
            vegetation_ratio=message.get("vegetation"),
            boulders_ratio=message.get("boulders"),
            ruins_ratio=message.get("ruins"),
            structures_ratio=message.get("structures"),
        )

    if action == "insight":
        metrics = message.get("metrics") or {}
        probability = float(message.get("probability", 0.0))
        return generate_insights(
            metrics=metrics,
            probability=probability,
            api_key=_normalize_key(message.get("apiKey") or ""),
            include_ai_insight=_to_bool(message.get("useAiInsight", True)),
            include_shap=_to_bool(message.get("includeShap", True)),
        )

    class_visibility = message.get("classVisibility") or {}

    return predict(
        image_path=message["image"],
        lat=float(message["lat"]),
        lon=float(message["lon"]),
        api_key=_normalize_key(message.get("apiKey") or ""),
        confidence=float(message.get("confidence", 0.25)),
        class_visibility={
            "vegetation": _to_bool(class_visibility.get("vegetation", True)),
            "ruins": _to_bool(class_visibility.get("ruins", True)),
            "structures": _to_bool(class_visibility.get("structures", True)),
            "boulders": _to_bool(class_visibility.get("boulders", True)),
            "others": _to_bool(class_visibility.get("others", True)),
        },
        use_ai_insight=_to_bool(message.get("useAiInsight", False)),
        include_shap=_to_bool(message.get("includeShap", False)),
        fast_mode=_to_bool(message.get("fastMode", False)),
    )


def main():
    print(json.dumps({"type": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        req_id = None
        try:
            message = json.loads(line)
            req_id = message.get("id")
            data = _handle_request(message)
            print(json.dumps({"id": req_id, "ok": True, "data": data}), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "id": req_id,
                        "ok": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
