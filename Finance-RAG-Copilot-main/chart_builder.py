"""
chart_builder.py - Financial charts from verified SEC data (line, bar, pie).

Charts use structured or parsed table data — not raw LLM guesses — for consistency.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

_FORBIDDEN_METRIC_TERMS = (
    "unearned",
    "deferred",
    "recognized revenue",
    "contract liability",
    "remaining performance",
)

_REVENUE_TERMS = ("revenue", "net sales", "sales", "turnover")
_YEAR_RE = re.compile(r"20\d{2}")
_CHART_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea", "#0891b2"]


def _period_sort_key(period: str) -> Tuple[int, int]:
    s = str(period).strip().upper()
    year_m = _YEAR_RE.search(s)
    year = int(year_m.group(0)) if year_m else 0
    quarter = 0
    qm = re.search(r"Q([1-4])", s)
    if qm:
        quarter = int(qm.group(1))
    elif re.fullmatch(r"20\d{2}", s.strip()):
        quarter = 5
    return (year, quarter)


def _looks_like_time(label: str) -> bool:
    s = str(label).strip().upper()
    return bool(_YEAR_RE.search(s) or re.search(r"Q[1-4]", s))


def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in ("", "-", "—", "n/a", "na", "none", "null", "not available")


def _parse_number(val: Any) -> Optional[float]:
    if _is_missing(val):
        return None
    s = str(val).replace(",", "").replace("$", "").strip()
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    if "-" in s and not s.startswith("-"):
        parts = nums
        if len(parts) >= 2:
            return (float(parts[0]) + float(parts[1])) / 2
    first = float(nums[0])
    if re.search(r"million|^\s*[\d.]+\s*m\s*$", s, re.I):
        return first / 1000.0
    return first


def _number_in_context(value: float, context: str, tolerance: float = 0.05) -> bool:
    if not context:
        return False

    # Check the value itself AND its millions equivalent (SEC filings store in $M)
    candidates = [value, round(value, 1), round(value, 2), value * 1000]

    ctx_clean = context.replace(",", "")

    for candidate in candidates:
        # Direct string match forms
        for fmt in (f"{candidate}", f"{candidate:,.1f}", f"{candidate:,.2f}", f"{int(candidate)}"):
            if fmt.replace(",", "") in ctx_clean:
                return True

    # Fuzzy numeric tolerance check against all numbers found in context
    ctx_nums = re.findall(r"[\d]+\.?\d*", ctx_clean)
    for cn in ctx_nums:
        try:
            c = float(cn)
            if c <= 0:
                continue
            # Match billions against billions, or billions against millions (*1000)
            if abs(c - value) / max(c, value) < tolerance:
                return True
            if abs(c - value * 1000) / max(c, value * 1000) < tolerance:
                return True
        except ValueError:
            continue
    return False


def validate_chart_data(chart_data: Dict[str, Any], context: str) -> Dict[str, Any]:
    out = dict(chart_data)
    points = chart_data.get("points") or []
    validated = []
    for pt in points:
        try:
            num = float(pt.get("value"))
        except (TypeError, ValueError):
            continue
        if _number_in_context(num, context):
            validated.append(pt)
    out["points"] = validated

    slices = chart_data.get("slices") or []
    val_slices = []
    for sl in slices:
        try:
            num = float(sl.get("value"))
        except (TypeError, ValueError):
            continue
        if _number_in_context(num, context):
            val_slices.append(sl)
    out["slices"] = val_slices
    return out


def _parse_markdown_table(text: str) -> Optional[pd.DataFrame]:
    lines = [ln.strip() for ln in text.splitlines() if "|" in ln]
    if len(lines) < 3:
        return None
    data_lines = [ln.strip("|") for ln in lines if not re.match(r"^[\s\|:-]+$", ln)]
    try:
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="|", skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return None


def _should_skip_column(col: str, user_query: str) -> bool:
    cl = col.lower()
    if any(x in cl for x in ("source", "ref", "change", "%")):
        return True
    q = user_query.lower()
    if "revenue" in q or "sales" in q:
        if any(term in cl for term in _FORBIDDEN_METRIC_TERMS):
            return True
    return False


def _pick_metric_column(columns: List[str], user_query: str) -> str:
    q = user_query.lower()
    scored = []
    for col in columns:
        cl = col.lower()
        score = 0
        if any(t in q for t in _REVENUE_TERMS) and any(t in cl for t in _REVENUE_TERMS):
            score += 10
        if "net income" in q and "net income" in cl:
            score += 10
        if any(term in cl for term in _FORBIDDEN_METRIC_TERMS):
            score -= 20
        scored.append((score, col))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else columns[0]


def chart_data_from_json_block(text: str) -> Optional[Dict[str, Any]]:
    patterns = [
        r"```chart-json\s*(\{.*?\})\s*```",
        r"```json\s*(\{.*?\})\s*```",
        r"(\{\s*\"metric_label\".*?\})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
    return None


def chart_data_from_markdown_table(text: str, user_query: str = "") -> Optional[Dict[str, Any]]:
    """Parse markdown tables into chart_data (trend, comparison, or breakdown)."""
    df = _parse_markdown_table(text)
    if df is None or len(df.columns) < 2:
        return None

    index_col = df.columns[0]
    value_cols = [c for c in df.columns[1:] if not _should_skip_column(c, user_query)]
    if not value_cols:
        return None

    index_vals = [str(v).strip() for v in df[index_col]]
    time_like = sum(1 for v in index_vals if _looks_like_time(v)) >= max(1, len(index_vals) // 2)

    # Wide comparison: Company | 2023 Revenue | 2024 Revenue
    if not time_like and len(value_cols) >= 1:
        series = []
        for _, row in df.iterrows():
            label = str(row[index_col]).strip()
            for col in value_cols:
                val = _parse_number(row[col])
                if val is not None:
                    series.append({
                        "period": f"{label}",
                        "value": val,
                        "group": col,
                    })
        if len(series) >= 2:
            q = user_query.lower()
            chart_type = "bar"
            if len(value_cols) == 1 and any(w in q for w in ("pie", "share", "breakdown", "portion", "split")):
                chart_type = "pie"
                for pt in series:
                    pt.pop("group", None)

            return {
                "chart_type": chart_type,
                "metric_label": _pick_metric_column(value_cols, user_query),
                "unit": "billions USD",
                "company": "",
                "points": series,
            }

    # Time series: Period | Metric
    primary = _pick_metric_column(value_cols, user_query)
    points = []
    for _, row in df.iterrows():
        period = str(row[index_col]).strip()
        val = _parse_number(row[primary])
        if val is not None:
            points.append({"period": period, "value": val})

    if len(points) < 2:
        return None

    q = user_query.lower()
    chart_type = "line" if time_like or "trend" in q else "bar"
    if any(w in q for w in ("pie", "share", "breakdown", "portion", "split")):
        if not time_like and 2 <= len(points) <= 8:
            chart_type = "pie"

    return {
        "chart_type": chart_type,
        "metric_label": primary,
        "unit": "billions USD",
        "company": "",
        "points": points,
    }


def chart_data_from_prose(text: str, user_query: str = "", intent: str = "") -> Optional[Dict[str, Any]]:
    """Last-resort parser for concise financial prose when no markdown table is present."""
    q = user_query.lower()
    body = str(text or "")
    body = re.sub(r"\[\d+\]", "", body)
    metric = "Total Revenue" if any(t in q for t in _REVENUE_TERMS) else "Amount"

    company_aliases = {
        "amazon": "Amazon",
        "apple": "Apple",
        "google": "Google",
        "alphabet": "Google",
        "meta": "Meta",
        "facebook": "Meta",
    }
    requested_companies = [
        label for key, label in company_aliases.items()
        if re.search(rf"\b{re.escape(key)}\b", q)
    ]
    requested_companies = list(dict.fromkeys(requested_companies))
    requested_years = list(dict.fromkeys(re.findall(r"20\d{2}", q)))

    points: List[Dict[str, Any]] = []
    segments = [
        seg.strip()
        for seg in re.split(r"(?<=[.!?])\s+|\n+", body)
        if seg.strip()
    ]

    def nearest_value(window: str, year: str) -> Optional[float]:
        direct_after_pattern = rf"\b{year}\b\s*(?:was|were|is|:|=|-)?\s*(\d+(?:\.\d+)?)\s*(?:billion|million|b|m)?"
        direct_after = re.search(direct_after_pattern, window, re.IGNORECASE)
        if direct_after and direct_after.group(1) != year:
            return float(direct_after.group(1))

        before = []
        before_pattern = rf"(\d+(?:\.\d+)?)\s*(?:billion|million|b|m)?[^.\n|]{{0,50}}\b{year}\b"
        for match in re.finditer(before_pattern, window, re.IGNORECASE):
            raw = match.group(1)
            if raw != year:
                before.append(float(raw))
        if before:
            return before[-1]

        after_pattern = rf"\b{year}\b[^.\n|]{{0,50}}?(\d+(?:\.\d+)?)\s*(?:billion|million|b|m)?"
        after = []
        for match in re.finditer(after_pattern, window, re.IGNORECASE):
            raw = match.group(1)
            if raw != year:
                after.append(float(raw))
        if after:
            return after[0]

        year_match = re.search(year, window)
        if not year_match:
            return None
        year_pos = year_match.start()
        candidates = []
        for num_match in re.finditer(r"\d+(?:\.\d+)?", window):
            raw = num_match.group(0)
            if raw == year:
                continue
            value = float(raw)
            distance = abs(num_match.start() - year_pos)
            candidates.append((distance, value))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    if intent != "trend" and len(requested_companies) > 1 and requested_years:
        for company in requested_companies:
            company_matches = list(re.finditer(re.escape(company), body, re.IGNORECASE))
            for year in requested_years:
                year_matches = list(re.finditer(year, body))
                candidate_segments = [
                    segment for segment in segments
                    if company.lower() in segment.lower() and year in segment
                ]
                if not candidate_segments:
                    candidates = company_matches + year_matches
                else:
                    candidates = []

                if not candidate_segments and not candidates:
                    continue
                best = None
                for segment in candidate_segments:
                    best = nearest_value(segment, year)
                    if best is not None:
                        break

                for match in candidates:
                    if best is not None:
                        break
                    start = max(0, match.start() - 140)
                    end = min(len(body), match.end() + 140)
                    window = body[start:end]
                    if company.lower() not in window.lower() or year not in window:
                        continue
                    best = nearest_value(window, year)
                    if best is not None:
                        break
                if best is not None:
                    points.append({"period": company, "group": year, "value": best})

    if len(points) >= 2:
        return {
            "chart_type": "bar",
            "metric_label": metric,
            "unit": "billions USD",
            "company": "",
            "points": points,
        }

    if (intent == "trend" or "trend" in q or "from" in q) and requested_years:
        for year in requested_years:
            for match in re.finditer(year, body):
                start = max(0, match.start() - 120)
                end = min(len(body), match.end() + 120)
                window = body[start:end]
                value = nearest_value(window, year)
                if value is not None:
                    points.append({"period": year, "value": value})
                    break

    if len(points) >= 2:
        return {
            "chart_type": "line",
            "metric_label": metric,
            "unit": "billions USD",
            "company": requested_companies[0] if requested_companies else "",
            "points": points,
        }

    # --- Pie / breakdown pass ---
    # If the user explicitly asked for a breakdown/pie, scan prose for labeled segments
    pie_keywords = ("pie", "breakdown", "segment", "split", "portion", "share", "composition")
    if any(w in q for w in pie_keywords):
        segment_pattern = re.compile(
            r"([A-Za-z][A-Za-z &/\-]+?)\s*[:\-–]\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:billion|B\b|million|M\b)?",
            re.IGNORECASE,
        )
        seg_points: List[Dict[str, Any]] = []
        for match in segment_pattern.finditer(body):
            label = match.group(1).strip()
            raw = float(match.group(2))
            # Skip year-like labels
            if re.match(r"^20\d{2}$", label) or re.match(r"^20\d{2}$", str(int(raw))):
                continue
            seg_points.append({"period": label, "value": raw})
        # Deduplicate by label
        seen_labels: set = set()
        unique_segs = []
        for sp in seg_points:
            if sp["period"] not in seen_labels:
                seen_labels.add(sp["period"])
                unique_segs.append(sp)
        if len(unique_segs) >= 2:
            return {
                "chart_type": "pie",
                "metric_label": metric,
                "unit": "billions USD",
                "company": requested_companies[0] if requested_companies else "",
                "points": unique_segs,
            }

    return None


def infer_chart_type(
    user_query: str,
    intent: str,
    chart_data: Dict[str, Any],
    preference: str = "auto",
) -> str:
    """Pick line, bar, or pie from query, intent, and data shape."""
    pref = (preference or "auto").lower()
    if pref in ("line", "bar", "pie"):
        return pref

    explicit = (chart_data.get("chart_type") or "auto").lower()
    if explicit in ("line", "bar", "pie"):
        return explicit

    q = user_query.lower()
    points = chart_data.get("points") or []
    slices = chart_data.get("slices") or []

    if any(w in q for w in ("pie", "donut", "share", "breakdown", "portion", "split", "composition")):
        if slices or (points and not _looks_like_time(points[0].get("period", ""))):
            return "pie"

    if intent == "comparison" or "compare" in q or chart_data.get("group"):
        return "bar"

    if intent == "trend" or any(w in q for w in ("trend", "over time", "growth", "yoy")):
        return "line"

    if points:
        if sum(1 for p in points if _looks_like_time(p.get("period", ""))) >= len(points) // 2:
            return "line"
        if "group" in (points[0] or {}):
            return "bar"
        if 2 <= len(points) <= 8 and not _looks_like_time(points[0].get("period", "")):
            if any(w in q for w in ("share", "percent", "%")):
                return "pie"
            return "bar"

    return "line"


def _layout(fig: go.Figure, title: str, dark_theme: bool, y_title: str = "") -> go.Figure:
    template = "plotly_dark" if dark_theme else "plotly_white"
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        template=template,
        height=440,
        margin=dict(t=70, b=60, l=60, r=40),
        font=dict(size=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    if y_title:
        fig.update_layout(yaxis_title=y_title)
    return fig


def _value_label(v: float, unit: str) -> str:
    if "billion" in unit.lower():
        return f"${v:.1f}B"
    return f"{v:.1f}"


def build_line_chart(chart_data: Dict[str, Any], *, dark_theme: bool = False) -> Optional[go.Figure]:
    points = [p for p in (chart_data.get("points") or []) if "group" not in p]
    if len(points) < 2:
        return None

    rows = []
    for pt in points:
        try:
            num = float(pt["value"])
        except (TypeError, ValueError, KeyError):
            continue
        rows.append({
            "period": str(pt["period"]),
            "value": num,
            "_order": _period_sort_key(pt["period"]),
        })
    if len(rows) < 2:
        return None

    df = pd.DataFrame(rows).sort_values("_order")
    metric = chart_data.get("metric_label", "Amount")
    company = chart_data.get("company", "")
    unit = chart_data.get("unit", "billions USD")
    title = f"{company} {metric} Over Time".strip() if company else f"{metric} Over Time"
    y_title = "Billions ($B)" if "billion" in unit.lower() else unit

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=df["value"],
            mode="lines+markers+text",
            text=[_value_label(v, unit) for v in df["value"]],
            textposition="top center",
            line=dict(width=4, color=_CHART_COLORS[0]),
            marker=dict(size=12, color=_CHART_COLORS[0]),
            connectgaps=False,
        )
    )
    fig.update_layout(xaxis_title="Time Period")
    return _layout(fig, title, dark_theme, y_title)


def build_bar_chart(chart_data: Dict[str, Any], *, dark_theme: bool = False) -> Optional[go.Figure]:
    points = chart_data.get("points") or []
    if len(points) < 2:
        return None

    has_groups = any(p.get("group") for p in points)
    metric = chart_data.get("metric_label", "Amount")
    company = chart_data.get("company", "")
    unit = chart_data.get("unit", "billions USD")
    title = f"{company} {metric} Comparison".strip() if company else f"{metric} Comparison"
    y_title = "Billions ($B)" if "billion" in unit.lower() else unit

    fig = go.Figure()

    if has_groups:
        groups: Dict[str, List[dict]] = {}
        for pt in points:
            g = pt.get("group", "Value")
            groups.setdefault(g, []).append(pt)
        for i, (gname, gpts) in enumerate(groups.items()):
            xs, ys, texts = [], [], []
            for pt in gpts:
                try:
                    v = float(pt["value"])
                except (TypeError, ValueError):
                    continue
                xs.append(str(pt["period"]))
                ys.append(v)
                texts.append(_value_label(v, unit))
            if xs:
                fig.add_trace(
                    go.Bar(
                        x=xs,
                        y=ys,
                        name=str(gname),
                        text=texts,
                        textposition="outside",
                        marker_color=_CHART_COLORS[i % len(_CHART_COLORS)],
                    )
                )
        fig.update_layout(barmode="group", xaxis_title="Category")
    else:
        xs, ys, texts = [], [], []
        for pt in sorted(points, key=lambda p: _period_sort_key(p.get("period", ""))):
            try:
                v = float(pt["value"])
            except (TypeError, ValueError):
                continue
            xs.append(str(pt["period"]))
            ys.append(v)
            texts.append(_value_label(v, unit))
        if len(xs) < 2:
            return None
        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys,
                text=texts,
                textposition="outside",
                marker_color=_CHART_COLORS[0],
                showlegend=False,
            )
        )
        fig.update_layout(xaxis_title="Category")

    return _layout(fig, title, dark_theme, y_title)


def build_pie_chart(chart_data: Dict[str, Any], *, dark_theme: bool = False) -> Optional[go.Figure]:
    slices = chart_data.get("slices") or []
    points = chart_data.get("points") or []

    labels, values = [], []
    if slices:
        for sl in slices:
            try:
                values.append(float(sl["value"]))
                labels.append(str(sl.get("label", sl.get("period", "?"))))
            except (TypeError, ValueError, KeyError):
                continue
    else:
        for pt in points:
            try:
                values.append(float(pt["value"]))
                labels.append(str(pt.get("period", pt.get("label", "?"))))
            except (TypeError, ValueError, KeyError):
                continue

    if len(values) < 2 or len(values) > 10:
        return None

    metric = chart_data.get("metric_label", "Share")
    company = chart_data.get("company", "")
    title = f"{company} {metric} Breakdown".strip() if company else f"{metric} Breakdown"

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.35,
            textinfo="label+percent",
            textposition="outside",
            marker=dict(colors=_CHART_COLORS),
        )
    )
    return _layout(fig, title, dark_theme)


def build_chart(
    chart_data: Dict[str, Any],
    user_query: str,
    intent: str,
    *,
    dark_theme: bool = False,
    chart_preference: str = "auto",
) -> Tuple[Optional[go.Figure], str]:
    """Build the best chart type; returns (figure, chart_type_used)."""
    ctype = infer_chart_type(user_query, intent, chart_data, chart_preference)

    builders = {
        "line": build_line_chart,
        "bar": build_bar_chart,
        "pie": build_pie_chart,
    }
    fig = builders[ctype](chart_data, dark_theme=dark_theme)

    # Fallback: line failed → try bar; pie failed → try bar
    if fig is None and ctype == "line":
        fig = build_bar_chart(chart_data, dark_theme=dark_theme)
        ctype = "bar" if fig else ctype
    if fig is None and ctype == "pie":
        fig = build_bar_chart(chart_data, dark_theme=dark_theme)
        ctype = "bar" if fig else ctype

    return fig, ctype


def trend_summary_plain(chart_data: Dict[str, Any]) -> str:
    points = chart_data.get("points") or []
    if len(points) < 2:
        return chart_data.get("notes") or ""

    ordered = sorted(
        [p for p in points if "group" not in p],
        key=lambda p: _period_sort_key(p["period"]),
    )
    if len(ordered) < 2:
        return chart_data.get("notes") or ""

    first, last = ordered[0], ordered[-1]
    v0, v1 = float(first["value"]), float(last["value"])
    metric = chart_data.get("metric_label", "Revenue")
    company = chart_data.get("company", "")
    who = f"{company}'s " if company else ""

    if v1 > v0:
        direction = "went **up**"
    elif v1 < v0:
        direction = "went **down**"
    else:
        direction = "stayed about the **same**"

    pct = abs((v1 - v0) / v0 * 100) if v0 else 0.0
    return (
        f"{who}{metric} {direction}: "
        f"**{first['period']}** was **${v0:.1f} billion**, "
        f"**{last['period']}** was **${v1:.1f} billion** "
        f"(about **{pct:.1f}%** change)."
    )


def format_trend_table(chart_data: Dict[str, Any]) -> str:
    metric = chart_data.get("metric_label", "Value")
    unit = chart_data.get("unit", "billions USD")
    company = chart_data.get("company", "")
    points = [p for p in (chart_data.get("points") or []) if "group" not in p]
    if not points:
        return chart_data.get("notes") or "The provided SEC filings do not contain enough data for this trend."

    title = f"{company} — {metric}" if company else metric
    lines = [
        f"### {title} ({unit})",
        "",
        f"| Period | {metric} ({unit.split()[0] if unit else 'USD'}) |",
        "|--------|" + "-" * max(12, len(metric)) + "|",
    ]
    for pt in sorted(points, key=lambda p: _period_sort_key(p["period"])):
        val = pt["value"]
        src = pt.get("source_id")
        cite = f" [{src}]" if src else ""
        lines.append(f"| {pt['period']} | {val:.1f}{cite} |")

    notes = chart_data.get("notes")
    if notes:
        lines.extend(["", notes])
    return "\n".join(lines)


def build_chart_from_result(
    result: Dict[str, Any],
    user_query: str,
    *,
    dark_theme: bool = False,
    chart_preference: str = "auto",
) -> Tuple[Optional[go.Figure], Optional[Dict[str, Any]], str]:
    """
    Build chart from orchestrate() result.
    Returns (figure, chart_data, chart_type_used).
    """
    chart_data = result.get("chart_data")
    context = result.get("context_text", "")
    intent = result.get("intent", "qa")

    def _validated(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not data:
            return None
        return validate_chart_data(data, context) if context else data

    def _has_chart_points(data: Optional[Dict[str, Any]]) -> bool:
        if not data:
            return False
        return len(data.get("points") or []) >= 2 or len(data.get("slices") or []) >= 2

    if chart_data:
        chart_data = _validated(chart_data)

    if not _has_chart_points(chart_data):
        ans = result.get("answer", "")
        ans_text = str(ans.get("long_answer", "")) if isinstance(ans, dict) else str(ans)
        fallback = chart_data_from_json_block(ans_text)
        if not fallback:
            fallback = chart_data_from_markdown_table(ans_text, user_query)
        if not fallback and _parse_markdown_table(ans_text) is None:
            fallback = chart_data_from_prose(ans_text, user_query, intent)
        if fallback:
            fallback = _validated(fallback)
            if _has_chart_points(fallback):
                chart_data = fallback

    if not chart_data:
        return None, None, "none"

    points = chart_data.get("points") or []
    slices = chart_data.get("slices") or []
    if len(points) < 2 and len(slices) < 2:
        return None, chart_data, "none"

    fig, ctype = build_chart(
        chart_data,
        user_query,
        intent,
        dark_theme=dark_theme,
        chart_preference=chart_preference,
    )
    return fig, chart_data, ctype


# Backward compatibility
def build_simple_chart(chart_data: Dict[str, Any], *, dark_theme: bool = False) -> Optional[go.Figure]:
    fig, _ = build_chart(chart_data, "", "trend", dark_theme=dark_theme)
    return fig
