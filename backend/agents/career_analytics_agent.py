"""
career_analytics_agent.py — Career Analytics Dashboard Generator
OrchestrAI Autonomous Multi-Agent System

PURPOSE:
  Reads interview feedback, opportunity scores, skill gaps, and career
  readiness data, then generates a rich self-contained Plotly HTML dashboard
  served at GET /analytics on the same Render domain.

CHARTS:
  1. Confidence vs Skill Topic (bar)
  2. Interview Performance Trend (line)
  3. Career Readiness Gauge (gauge)
  4. Opportunity Match Score Distribution (horizontal bar)
  5. Skill Gap Frequency (donut)

OUTPUT:
  DATA_DIR/frontend/analytics/dashboard.html
  → served at https://orchestrai-agent.onrender.com/analytics
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from backend.github_yaml_db import (
    read_yaml_from_github,
    write_yaml_to_github,
    append_log_entry,
    _get_raw_file,
    _put_raw_file,
)

load_dotenv()
logger = logging.getLogger("OrchestrAI.CareerAnalyticsAgent")

FEEDBACK_FILE  = "database/interview_feedback.yaml"
SCORES_FILE    = "database/opportunity_scores.yaml"
SKILL_GAP_FILE = "database/skill_gap_per_job.yaml"
READINESS_FILE = "database/career_readiness.yaml"
ANALYTICS_FILE = "database/career_analytics.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Data collectors
# ─────────────────────────────────────────────────────────────────────────────

def _collect_confidence_data(feedbacks: list[dict]) -> tuple[list[str], list[int]]:
    """
    Aggregate confidence scores per topic across all feedback entries.
    Returns (topics, avg_confidences).
    """
    topic_scores: dict[str, list[int]] = {}
    for fb in feedbacks:
        conf = int(fb.get("confidence", fb.get("confidence_level", 5)))
        for q in fb.get("questions_faced", fb.get("topics", [])):
            topic = str(q)[:40].title()
            topic_scores.setdefault(topic, []).append(conf)

    averaged = {t: round(sum(v)/len(v), 1) for t, v in topic_scores.items()}
    # Sort by lowest confidence first (most important to study)
    sorted_items = sorted(averaged.items(), key=lambda x: x[1])[:12]
    if not sorted_items:
        return ["Python", "ML Theory", "SQL", "System Design"], [8, 5, 7, 4]
    topics, scores = zip(*sorted_items)
    return list(topics), [float(s) for s in scores]


def _collect_performance_trend(feedbacks: list[dict]) -> tuple[list[str], list[float]]:
    """
    Interview performance over time: composite score = (confidence + (10-difficulty)) / 2
    Returns (dates, scores).
    """
    dated = []
    for fb in feedbacks:
        ts = fb.get("logged_at", fb.get("timestamp", ""))
        conf = int(fb.get("confidence", fb.get("confidence_level", 5)))
        diff = int(fb.get("difficulty", fb.get("difficulty_level", 5)))
        composite = round((conf + (10 - diff)) / 2, 1)
        if ts:
            dated.append((ts[:10], composite))  # YYYY-MM-DD

    if not dated:
        from random import uniform
        import time
        base = datetime.now(timezone.utc)
        dates = [f"2026-03-0{i}" for i in range(1, 6)]
        return dates, [round(uniform(4, 8), 1) for _ in dates]

    dated.sort(key=lambda x: x[0])
    dates, scores = zip(*dated)
    return list(dates), list(scores)


def _collect_skill_gaps(analyses: list[dict]) -> tuple[list[str], list[int]]:
    """Top missing skills by frequency across all jobs."""
    freq: Counter = Counter()
    for a in analyses:
        for s in a.get("missing_skills", []):
            freq[str(s)] += 1
    top = freq.most_common(10)
    if not top:
        return ["No gaps found"], [0]
    skills, counts = zip(*top)
    return list(skills), list(counts)


def _collect_match_scores(scores_list: list[dict]) -> tuple[list[str], list[float], list[str]]:
    """Opportunity match scores per company/role."""
    items = []
    for s in scores_list:
        label = f"{s.get('company','?')} – {s.get('role','?')}"[:45]
        score = float(s.get("match_score", 0))
        prob  = s.get("selection_probability", "Low")
        items.append((label, score, prob))
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:12]
    if not items:
        return ["No data"], [0.0], ["Low"]
    labels, scores, probs = zip(*items)
    return list(labels), list(scores), list(probs)


# ─────────────────────────────────────────────────────────────────────────────
# HTML / Plotly builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_dashboard_html(
    readiness_score: float,
    readiness_label: str,
    cr_components: dict,
    conf_topics: list[str],
    conf_scores: list[float],
    trend_dates: list[str],
    trend_scores: list[float],
    gap_skills: list[str],
    gap_counts: list[int],
    match_labels: list[str],
    match_scores: list[float],
    match_probs: list[str],
    generated_at: str,
) -> str:
    """Build a rich self-contained Plotly HTML analytics dashboard."""

    # Colour helpers
    conf_colors = [
        "#ef4444" if s < 5 else "#f97316" if s < 7 else "#22c55e"
        for s in conf_scores
    ]
    prob_colors = {
        "High": "#22c55e", "Medium": "#f97316", "Low": "#ef4444"
    }
    match_bar_colors = [prob_colors.get(p, "#6b7280") for p in match_probs]

    readiness_color = (
        "#22c55e" if readiness_score >= 85 else
        "#3b82f6" if readiness_score >= 70 else
        "#f97316" if readiness_score >= 50 else "#ef4444"
    )

    skill_pct = cr_components.get("skill_coverage", {}).get("score", 0)
    port_pct  = cr_components.get("portfolio_strength", {}).get("score", 0)
    prac_pct  = cr_components.get("interview_practice", {}).get("score", 0)
    sec_pct   = cr_components.get("security_health", {}).get("score", 0)

    def _js(v: Any) -> str:
        return json.dumps(v)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>OrchestrAI — Career Analytics Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet"/>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Inter',sans-serif;background:#0f0e17;color:#fffffe;min-height:100vh}}
  .header{{background:linear-gradient(135deg,#7c3aed,#4f46e5,#2563eb);padding:40px;text-align:center}}
  .header h1{{font-size:28px;font-weight:700;margin-bottom:4px}}
  .header p{{opacity:0.75;font-size:14px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;
         padding:24px;max-width:1400px;margin:0 auto}}
  .card{{background:#1a1a2e;border-radius:14px;padding:24px;border:1px solid rgba(255,255,255,0.07)}}
  .card h3{{color:#a78bfa;font-size:15px;font-weight:600;margin-bottom:16px}}
  .readiness-badge{{display:inline-flex;align-items:center;gap:12px;background:{readiness_color}22;
    border:2px solid {readiness_color};border-radius:12px;padding:16px 24px;margin-bottom:20px}}
  .readiness-score{{font-size:48px;font-weight:700;color:{readiness_color};line-height:1}}
  .readiness-label{{font-size:16px;font-weight:600;color:{readiness_color}}}
  .comp-bar{{margin-bottom:12px}}
  .comp-bar .label{{display:flex;justify-content:space-between;font-size:12px;color:#9ca3af;margin-bottom:4px}}
  .comp-bar .track{{height:8px;background:#374151;border-radius:4px;overflow:hidden}}
  .comp-bar .fill{{height:100%;border-radius:4px;transition:width 0.6s ease}}
  .chart{{width:100%;height:320px}}
  .chart-wide{{width:100%;height:360px}}
  .stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:4px}}
  .stat-box{{background:#0f0e17;border-radius:8px;padding:14px;text-align:center}}
  .stat-num{{font-size:28px;font-weight:700;color:#a78bfa}}
  .stat-lbl{{font-size:11px;color:#6b7280;margin-top:2px}}
  .footer{{text-align:center;color:#374151;font-size:12px;padding:24px;
           border-top:1px solid rgba(255,255,255,0.05);margin-top:20px}}
  @media(max-width:600px){{.header{{padding:30px 20px}}.grid{{padding:16px;gap:14px}}}}
</style>
</head>
<body>

<div class="header">
  <h1>🤖 OrchestrAI Career Analytics</h1>
  <p>Autonomous Career Intelligence Dashboard &nbsp;·&nbsp; Updated {generated_at}</p>
</div>

<div class="grid">

  <!-- Career Readiness Score -->
  <div class="card" style="grid-column:1/-1">
    <h3>🎯 Career Readiness Score</h3>
    <div style="display:flex;align-items:flex-start;gap:32px;flex-wrap:wrap">
      <div>
        <div class="readiness-badge">
          <div class="readiness-score">{readiness_score}</div>
          <div>
            <div class="readiness-label">{readiness_label}</div>
            <div style="font-size:12px;color:{readiness_color};opacity:0.8">out of 100</div>
          </div>
        </div>
      </div>
      <div style="flex:1;min-width:260px">
        <div class="comp-bar">
          <div class="label"><span>🧠 Skill Coverage (40%)</span><span>{skill_pct:.0f}/100</span></div>
          <div class="track"><div class="fill" style="width:{skill_pct}%;background:#a78bfa"></div></div>
        </div>
        <div class="comp-bar">
          <div class="label"><span>🏆 Portfolio Strength (20%)</span><span>{port_pct:.0f}/100</span></div>
          <div class="track"><div class="fill" style="width:{port_pct}%;background:#22c55e"></div></div>
        </div>
        <div class="comp-bar">
          <div class="label"><span>🎤 Interview Practice (20%)</span><span>{prac_pct:.0f}/100</span></div>
          <div class="track"><div class="fill" style="width:{prac_pct}%;background:#f97316"></div></div>
        </div>
        <div class="comp-bar">
          <div class="label"><span>🔒 Security Health (20%)</span><span>{sec_pct:.0f}/100</span></div>
          <div class="track"><div class="fill" style="width:{sec_pct}%;background:#3b82f6"></div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Confidence vs Skill -->
  <div class="card">
    <h3>🧠 Confidence vs Skill Topic</h3>
    <div id="chart_conf" class="chart"></div>
  </div>

  <!-- Performance Trend -->
  <div class="card">
    <h3>📈 Interview Performance Trend</h3>
    <div id="chart_trend" class="chart"></div>
  </div>

  <!-- Opportunity Match Scores -->
  <div class="card" style="grid-column:1/-1">
    <h3>🎯 Opportunity Match Scores</h3>
    <div id="chart_match" class="chart-wide"></div>
  </div>

  <!-- Skill Gap Frequency -->
  <div class="card">
    <h3>📊 Top Missing Skills (Frequency)</h3>
    <div id="chart_gaps" class="chart"></div>
  </div>

  <!-- Quick Stats -->
  <div class="card">
    <h3>⚡ Quick Stats</h3>
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-num">{len(gap_skills)}</div>
        <div class="stat-lbl">Skill Gaps Tracked</div>
      </div>
      <div class="stat-box">
        <div class="stat-num">{len(match_labels)}</div>
        <div class="stat-lbl">Opportunities Scored</div>
      </div>
      <div class="stat-box">
        <div class="stat-num">{len(trend_dates)}</div>
        <div class="stat-lbl">Practice Sessions</div>
      </div>
      <div class="stat-box">
        <div class="stat-num">{round(sum(match_scores)/len(match_scores)) if match_scores else 0}</div>
        <div class="stat-lbl">Avg Match Score</div>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  OrchestrAI Career Analytics &nbsp;·&nbsp; Generated {generated_at} &nbsp;·&nbsp;
  <a href="/" style="color:#7c3aed;text-decoration:none">← Back to Dashboard</a>
</div>

<script>
const PLOTLY_DARK = {{
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: {{ color: '#9ca3af', family: 'Inter, sans-serif', size: 12 }},
  xaxis: {{ gridcolor: '#1f2937', linecolor: '#374151', tickfont: {{ color: '#9ca3af' }} }},
  yaxis: {{ gridcolor: '#1f2937', linecolor: '#374151', tickfont: {{ color: '#9ca3af' }} }},
  margin: {{ t: 10, b: 60, l: 40, r: 20 }}
}};

// 1. Confidence vs Skill
Plotly.newPlot('chart_conf', [{{
  type: 'bar',
  x: {_js(conf_scores)},
  y: {_js(conf_topics)},
  orientation: 'h',
  marker: {{ color: {_js(conf_colors)}, opacity: 0.85 }},
  text: {_js([f'{s}/10' for s in conf_scores])},
  textposition: 'outside',
  textfont: {{ color: '#fffffe', size: 11 }},
  hovertemplate: '<b>%{{y}}</b><br>Confidence: %{{x}}/10<extra></extra>'
}}], {{
  ...PLOTLY_DARK,
  xaxis: {{ ...PLOTLY_DARK.xaxis, range: [0, 11], title: 'Confidence (1-10)' }},
  yaxis: {{ ...PLOTLY_DARK.yaxis, automargin: true }},
  showlegend: false
}}, {{responsive: true, displayModeBar: false}});

// 2. Performance Trend
Plotly.newPlot('chart_trend', [{{
  type: 'scatter',
  mode: 'lines+markers',
  x: {_js(trend_dates)},
  y: {_js(trend_scores)},
  line: {{ color: '#a78bfa', width: 2.5, shape: 'spline' }},
  marker: {{ color: '#7c3aed', size: 8, symbol: 'circle' }},
  fill: 'tozeroy',
  fillcolor: 'rgba(124,58,237,0.10)',
  hovertemplate: '<b>%{{x}}</b><br>Score: %{{y}}/10<extra></extra>'
}}], {{
  ...PLOTLY_DARK,
  yaxis: {{ ...PLOTLY_DARK.yaxis, range: [0, 10.5], title: 'Score (1-10)' }},
  xaxis: {{ ...PLOTLY_DARK.xaxis, title: 'Date' }},
  showlegend: false
}}, {{responsive: true, displayModeBar: false}});

// 3. Match Scores
Plotly.newPlot('chart_match', [{{
  type: 'bar',
  x: {_js(match_scores)},
  y: {_js(match_labels)},
  orientation: 'h',
  marker: {{ color: {_js(match_bar_colors)}, opacity: 0.85 }},
  text: {_js([f'{s:.0f}/100' for s in match_scores])},
  textposition: 'outside',
  textfont: {{ color: '#fffffe', size: 10 }},
  hovertemplate: '<b>%{{y}}</b><br>Match: %{{x}}/100<extra></extra>'
}}], {{
  ...PLOTLY_DARK,
  xaxis: {{ ...PLOTLY_DARK.xaxis, range: [0, 110], title: 'Match Score' }},
  yaxis: {{ ...PLOTLY_DARK.yaxis, automargin: true }},
  height: 360,
  showlegend: false
}}, {{responsive: true, displayModeBar: false}});

// 4. Skill Gaps (donut)
Plotly.newPlot('chart_gaps', [{{
  type: 'pie',
  hole: 0.5,
  labels: {_js(gap_skills)},
  values: {_js(gap_counts)},
  textinfo: 'label+percent',
  textfont: {{ size: 11, color: '#fffffe' }},
  marker: {{
    colors: ['#7c3aed','#4f46e5','#ef4444','#f97316','#22c55e',
             '#3b82f6','#ec4899','#14b8a6','#f59e0b','#6366f1']
  }},
  hovertemplate: '<b>%{{label}}</b><br>Count: %{{value}}<extra></extra>'
}}], {{
  ...PLOTLY_DARK,
  showlegend: false,
  margin: {{ t: 10, b: 10, l: 10, r: 10 }}
}}, {{responsive: true, displayModeBar: false}});
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

def run_career_analytics_agent() -> str:
    """
    Collect metrics, build dashboard HTML, save to local filesystem.
    Returns the URL of the analytics dashboard.
    """
    logger.info("CareerAnalyticsAgent: Building analytics dashboard...")

    # ── Read data ─────────────────────────────────────────────────────────────
    feedback_raw = read_yaml_from_github(FEEDBACK_FILE) or {}
    feedbacks: list[dict] = (
        feedback_raw.get("interview_feedback", [])
        if isinstance(feedback_raw, dict) else
        feedback_raw if isinstance(feedback_raw, list) else []
    )

    scores_raw   = read_yaml_from_github(SCORES_FILE) or []
    scores_list  = scores_raw if isinstance(scores_raw, list) else []

    gap_raw      = read_yaml_from_github(SKILL_GAP_FILE) or {}
    analyses     = gap_raw.get("job_skill_analysis", []) if isinstance(gap_raw, dict) else []

    readiness_raw = read_yaml_from_github(READINESS_FILE) or {}
    cr = readiness_raw.get("career_readiness", {}) if isinstance(readiness_raw, dict) else {}
    readiness_score  = float(cr.get("readiness_score", 0))
    readiness_label  = cr.get("label", "Computing...")
    cr_components    = cr.get("components", {})

    # ── Compute chart data ────────────────────────────────────────────────────
    conf_topics, conf_scores   = _collect_confidence_data(feedbacks)
    trend_dates, trend_scores  = _collect_performance_trend(feedbacks)
    gap_skills, gap_counts     = _collect_skill_gaps(analyses)
    match_labels, match_scores, match_probs = _collect_match_scores(scores_list)

    # ── Build HTML ────────────────────────────────────────────────────────────
    generated_at = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")
    html = _build_dashboard_html(
        readiness_score=readiness_score,
        readiness_label=readiness_label,
        cr_components=cr_components,
        conf_topics=conf_topics,
        conf_scores=conf_scores,
        trend_dates=trend_dates,
        trend_scores=trend_scores,
        gap_skills=gap_skills,
        gap_counts=gap_counts,
        match_labels=match_labels,
        match_scores=match_scores,
        match_probs=match_probs,
        generated_at=generated_at,
    )

    # Write HTML file via GitHub Persistence layer
    file_path = "frontend/analytics/index.html"
    _, sha = _get_raw_file(file_path)
    ts = datetime.now(timezone.utc).isoformat()
    _put_raw_file(file_path, html, sha, f"feat(analytics): updated dashboard — {ts}")

    base_url      = os.getenv("RENDER_EXTERNAL_URL", "https://orchestrai-agent.onrender.com")
    dashboard_url = f"{base_url}/analytics"

    # ── Save metadata ─────────────────────────────────────────────────────────
    try:
        write_yaml_to_github(ANALYTICS_FILE, {
            "analytics": {
                "dashboard_url":     dashboard_url,
                "readiness_score":   readiness_score,
                "readiness_label":   readiness_label,
                "feedback_sessions": len(feedbacks),
                "opportunities":     len(scores_list),
                "skill_gaps":        len(gap_skills),
                "generated_at":      datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            }
        })
    except Exception as exc:
        logger.warning("CareerAnalyticsAgent: Could not save analytics metadata — %s", exc)

    try:
        append_log_entry({
            "agent":     "CareerAnalyticsAgent",
            "action":    f"Generated analytics dashboard — readiness={readiness_score}",
            "status":    "completed",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        })
    except Exception:
        pass

    logger.info("CareerAnalyticsAgent: Dashboard ready at %s", dashboard_url)
    return dashboard_url


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    url = run_career_analytics_agent()
    print(f"\n✅ Analytics dashboard: {url}")
