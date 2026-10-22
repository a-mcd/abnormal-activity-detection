

def serialize_reference_stats(reference_stats: dict) -> dict:
    if not reference_stats:
        return {}

    if not isinstance(reference_stats, dict):
        raise TypeError(
            f"serialize_reference_stats expected dict, got {type(reference_stats).__name__}"
        )

    # Base case: one reference-stats object
    if "event_vocab" in reference_stats:
        transition_prob = reference_stats.get("transition_prob", {})

        return {
            "event_vocab": sorted(list(reference_stats.get("event_vocab", []))),
            # Sorted so the saved file is the same on every run (the vocabularies are sets)
            "bigram_vocab": [list(x) for x in sorted(reference_stats.get("bigram_vocab", []))],
            "trigram_vocab": [list(x) for x in sorted(reference_stats.get("trigram_vocab", []))],
            "transition_prob": {
                "|||".join(bg): float(prob)
                for bg, prob in transition_prob.items()
            },
        }

    # Recursive case: nested dict such as {"t1": {...}, "t2": {...}}
    return {
        key: serialize_reference_stats(value)
        for key, value in reference_stats.items()
    }


def deserialize_reference_stats(reference_stats_json: dict) -> dict:
    if not reference_stats_json:
        return {}

    if not isinstance(reference_stats_json, dict):
        raise TypeError(
            f"deserialize_reference_stats expected dict, got {type(reference_stats_json).__name__}"
        )

    if "event_vocab" in reference_stats_json:
        transition_prob_json = reference_stats_json.get("transition_prob", {})

        return {
            "event_vocab": set(reference_stats_json.get("event_vocab", [])),
            "bigram_vocab": {
                tuple(x) for x in reference_stats_json.get("bigram_vocab", [])
            },
            "trigram_vocab": {
                tuple(x) for x in reference_stats_json.get("trigram_vocab", [])
            },
            "transition_prob": {
                tuple(k.split("|||")): float(v)
                for k, v in transition_prob_json.items()
            },
        }

    return {
        key: deserialize_reference_stats(value)
        for key, value in reference_stats_json.items()
    }