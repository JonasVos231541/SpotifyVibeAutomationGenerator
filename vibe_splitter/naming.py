"""
Cluster naming and energy/mood scoring.

Provides:
  - ``score_axis``          — tag-based 0-1 scoring for energy and mood
  - ``genre_name_from_tags`` — match tags against genre rules
  - ``name_all_clusters``   — full naming pipeline with cross-cluster distinctiveness
  - ``generate_ai_names``   — query Ollama for creative names
  - ``generate_vibe_categories`` — ask Ollama to suggest vibe categories for clustering guidance
"""
import logging, requests
from collections import Counter
from . import config

log = logging.getLogger("splitter.naming")


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_axis(tag_counter, pos_set, neg_set):
    """Score a cluster on one axis (0 = negative, 0.5 = neutral, 1 = positive)."""
    score, hits = 0.5, 0
    for tag, count in tag_counter.items():
        w = min(count / 100.0, 1.0)
        if tag in pos_set:
            score += 0.4 * w; hits += 1
        elif tag in neg_set:
            score -= 0.4 * w; hits += 1
    return max(0.0, min(1.0, score if hits else 0.5))


def energy_band(s):
    return "high" if s >= 0.58 else ("mid" if s >= 0.42 else "low")

def mood_band(s):
    return "bright" if s >= 0.57 else ("neutral" if s >= 0.43 else "dark")


# ─── Genre matching ───────────────────────────────────────────────────────────

def genre_name_from_tags(top_tags):
    """Match *top_tags* against ``GENRE_RULES``.  Returns ``(name, desc)`` or ``None``."""
    tag_set = set(top_tags)
    best_key, best_desc, best_matches = None, "", 0
    for trigger_set, min_matches, genre_key, desc in config.GENRE_RULES:
        matches = len(trigger_set & tag_set)
        if matches >= min_matches and matches > best_matches:
            best_matches = matches
            best_key     = genre_key
            best_desc    = desc
    return (best_key, best_desc) if best_key else None


# ─── Ollama AI naming ─────────────────────────────────────────────────────────

def generate_ai_names(top_tags, energy, mood):
    """Return list of up to 5 name suggestions, or None if Ollama fails."""
    if not config.ENABLE_AI_NAMES:
        return None
    prompt = f"""You are a music curator. Based on these tags: {', '.join(top_tags[:8])},
energy level: {energy}, mood: {mood},
generate 5 creative playlist names (max 3 words each) that capture the vibe.
Return only the names, one per line."""
    try:
        resp = requests.post(config.OLLAMA_URL, json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }, timeout=30)
        if resp.status_code == 200:
            text = resp.json().get("response", "")
            # Split lines and clean
            names = [line.strip().strip('"').strip() for line in text.split('\n') if line.strip()]
            return names[:5]
    except Exception as e:
        log.warning(f"Ollama request failed: {e}")
    return None


# ─── LLM Guidance: Generate vibe categories from a sample ────────────────────

def generate_vibe_categories(records_sample, min_cats=5, max_cats=10):
    """
    Ask Llama to suggest vibe categories based on a sample of tracks.
    Returns a list of category name strings.
    """
    if not records_sample:
        return []
    
    # Build a summary of the sample: for each track, show top tags and artist
    sample_text = ""
    for i, r in enumerate(records_sample[:30]):  # limit to 30 to avoid huge prompts
        tags = ", ".join(r.get("tags", [])[:5])
        artist = r.get("artist", "Unknown")
        sample_text += f"{i+1}. Artist: {artist}, Tags: {tags}\n"
    
    prompt = f"""You are a music curator analyzing a diverse set of tracks.
Here is a sample of {len(records_sample)} tracks with their top tags and artists:

{sample_text}

Based on this sample, identify between {min_cats} and {max_cats} distinct vibe categories that capture the main moods, genres, or listening contexts.
Each category should be a short phrase (2-4 words) that describes a vibe (e.g., "Late Night Drive", "Morning Coffee", "Gym Bangers", "Rainy Day Melancholy").

Return ONLY the category names, one per line, with no numbering or extra text."""
    
    try:
        resp = requests.post(config.OLLAMA_URL, json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }, timeout=30)
        if resp.status_code == 200:
            text = resp.json().get("response", "")
            # Split lines and clean
            categories = [line.strip().strip('"').strip() for line in text.split('\n') if line.strip()]
            # Limit to max_cats
            categories = categories[:max_cats]
            if len(categories) < min_cats:
                # If Llama returned too few, pad with generic ones
                categories.extend(["Mixed Vibes", "Energy Boost", "Chill Out"][:min_cats - len(categories)])
            return categories
    except Exception as e:
        log.warning(f"Llama category generation failed: {e}")
    return []


# ─── Name construction ────────────────────────────────────────────────────────

def _pick_name_tags(top_tags, n=3, avoid=None):
    """Pick the most distinctive tags for naming, optionally skipping *avoid*."""
    avoid = avoid or set()
    picked = []
    for tag in top_tags:
        if tag.lower() not in config.SKIP_NAME_TAGS and tag.lower() not in avoid and len(tag) > 2:
            picked.append(tag.title())
        if len(picked) >= n:
            break
    return picked or [t.title() for t in top_tags[:2]]


def _generate_cluster_name(top_tags, e_band, m_band, genre_key=None, shared_tags=None):
    avoid = shared_tags or set()
    if genre_key:
        tags = _pick_name_tags(top_tags, n=2, avoid=avoid)
        qualifier = f" · {tags[0]}" if tags and tags[0].lower() != genre_key.lower() else ""
        return f"{genre_key}{qualifier}"
    tags = _pick_name_tags(top_tags, n=3, avoid=avoid)
    if len(tags) >= 2:
        return f"{tags[0]} × {tags[1]}"
    elif tags:
        flavour = {"high": "Energy", "low": "Chill", "mid": ""}.get(e_band, "")
        return f"{tags[0]} {flavour}" if flavour else tags[0]
    return "Mixed Vibes"


def _generate_description(top_tags, e_band, m_band):
    tags = [t.title() for t in top_tags[:6]]
    energy_desc = {"high": "high energy", "mid": "mid-energy", "low": "laid-back"}.get(e_band, "")
    mood_desc   = {"bright": "uplifting", "neutral": "balanced", "dark": "moody"}.get(m_band, "")
    return f"{energy_desc.title()}, {mood_desc} — {', '.join(tags[:4])}"


def _generate_pinterest_prompts(top_tags, name):
    tags = _pick_name_tags(top_tags, n=4)
    prompts = []
    if len(tags) >= 2:
        prompts.append(f"{tags[0]} {tags[1]} aesthetic playlist cover dark")
    prompts.append(f"{name} aesthetic playlist cover moody")
    prompts.append(f"{' '.join(tags[:3])} music aesthetic photography")
    return prompts[:3]


# ─── Main naming pipeline ─────────────────────────────────────────────────────

def name_all_clusters(cluster_meta):
    """
    Name every cluster with cross-cluster distinctiveness.

    Tags appearing in >50% of clusters are deprioritised for naming so that
    each name reflects what makes a cluster *unique*.

    Returns ``{key: (name, desc, e_score, m_score, pinterest_prompts, ai_suggestions)}``.
    """
    # Find shared (non-distinctive) tags
    tag_cluster_count = Counter()
    n_clusters = len(cluster_meta)
    for m in cluster_meta.values():
        for tag in set(m.get("top_tags", [])):
            tag_cluster_count[tag] += 1
    shared_tags = {tag for tag, cnt in tag_cluster_count.items()
                   if cnt > max(2, n_clusters * 0.5)}

    seen, result = set(), {}
    for idx, (key, m) in enumerate(cluster_meta.items()):
        top_tags = m.get("top_tags", [])
        e_b      = m["quad"][0]
        m_b      = m["quad"][1]
        e_score  = round(m["e"], 2)
        m_score  = round(m.get("m", 0.5), 2)

        genre_match = genre_name_from_tags(top_tags)
        genre_key   = genre_match[0] if genre_match else None

        name = _generate_cluster_name(top_tags, e_b, m_b, genre_key, shared_tags)
        if name in seen:
            extra = _pick_name_tags(top_tags[3:], n=1, avoid=shared_tags)
            name = f"{name} · {extra[0]}" if extra else f"{name} {idx+1}"
        if name in seen:
            name = f"{name} {idx+1}"
        seen.add(name)

        desc    = genre_match[1] if genre_match else _generate_description(top_tags, e_b, m_b)
        prompts = _generate_pinterest_prompts(top_tags, name)

        # AI name suggestions
        ai_names = generate_ai_names(top_tags, e_b, m_b)

        result[key] = (name, desc, e_score, m_score, prompts, ai_names)

    return result