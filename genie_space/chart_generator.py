"""
Auto-visualization module for Genie Space responses.

Uses an LLM to evaluate whether tabular data warrants a chart,
select the chart type, and generate a plotly figure.

Auth: Service Principal (same as LLM insights) — 'model-serving'
scope is not available for Databricks Apps OBO tokens.
"""

import json
import logging
import os
from typing import Optional, Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a data visualization expert. Given a user's question, a SQL description, \
and a sample of the resulting data, determine whether a chart would effectively \
communicate the answer. If yes, provide a chart specification.

Respond ONLY with valid JSON. No markdown, no code fences, no explanation.

If a chart is NOT appropriate (e.g., single-row result, text-heavy data, \
no clear visual relationship), respond exactly: {"chart": false}

If a chart IS appropriate, respond with:
{
  "chart": true,
  "chart_type": "<bar|line|scatter|pie|histogram|area>",
  "x": "<column_name>",
  "y": "<column_name or list of column_names>",
  "color": "<column_name or null>",
  "title": "<descriptive chart title>",
  "orientation": "<v|h>",
  "agg_func": "<sum|mean|count|null>"
}

Rules:
- chart_type must be one of: bar, line, scatter, pie, histogram, area
- x, y, color must be exact column names from the data
- For pie charts: use x as names and y as values
- For histograms: only x is required
- orientation defaults to "v" (vertical); use "h" for horizontal bar charts
- agg_func: if the data needs aggregation before plotting, specify the function; \
otherwise null
- Prefer bar charts for categorical comparisons
- Prefer line charts for time series
- Prefer pie charts only when showing parts of a whole (<=7 categories)
- Never suggest a chart for a single row of data
- Never suggest a bar chart when x has >50 unique values"""


def _build_user_prompt(
    df: pd.DataFrame,
    user_question: str,
    sql_description: Optional[str],
    space_instructions: Optional[str],
) -> str:
    parts = [
        f"User question: {user_question}",
        f"SQL description: {sql_description or 'N/A'}",
        "",
        f"Data columns: {list(df.columns)}",
        f"Data types: {dict(df.dtypes.astype(str))}",
        f"Row count: {len(df)}",
        f"Sample (first 5 rows):\n{df.head(5).to_csv(index=False)}",
    ]
    if space_instructions:
        parts.append(f"Context: {space_instructions}")
    return "\n".join(parts)


def _get_chart_spec(
    df: pd.DataFrame,
    user_question: str,
    sql_description: Optional[str],
    space_instructions: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Call LLM to get chart specification. Returns None if no chart."""
    endpoint_name = os.getenv("VISUALIZATION_ENDPOINT_NAME") or os.getenv(
        "SERVING_ENDPOINT_NAME"
    )
    if not endpoint_name:
        logger.warning("[chart_generator] No serving endpoint configured")
        return None

    user_prompt = _build_user_prompt(
        df, user_question, sql_description, space_instructions
    )

    try:
        client = WorkspaceClient()  # SP auth
        response = client.serving_endpoints.query(
            endpoint_name,
            messages=[
                ChatMessage(content=_SYSTEM_PROMPT, role=ChatMessageRole.SYSTEM),
                ChatMessage(content=user_prompt, role=ChatMessageRole.USER),
            ],
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if the model wraps them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        spec = json.loads(raw)

        if not spec.get("chart"):
            logger.info("[chart_generator] LLM decided no chart needed")
            return None

        # Validate column references
        for key in ("x", "color"):
            val = spec.get(key)
            if val and val != "null" and val not in df.columns:
                logger.warning(
                    f"[chart_generator] Column '{val}' not in DataFrame, skipping chart"
                )
                return None

        y_val = spec.get("y")
        if isinstance(y_val, str):
            if y_val not in df.columns:
                logger.warning(
                    f"[chart_generator] Column '{y_val}' not in DataFrame, skipping chart"
                )
                return None
        elif isinstance(y_val, list):
            for col in y_val:
                if col not in df.columns:
                    logger.warning(
                        f"[chart_generator] Column '{col}' not in DataFrame, skipping chart"
                    )
                    return None

        logger.info(
            f"[chart_generator] LLM recommended: {spec.get('chart_type')} chart"
        )
        return spec

    except json.JSONDecodeError as e:
        logger.warning(f"[chart_generator] LLM returned invalid JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"[chart_generator] LLM call failed: {e}")
        return None


def _build_figure(df: pd.DataFrame, spec: Dict[str, Any]) -> Optional[go.Figure]:
    """Build a plotly figure from an LLM-generated spec."""
    chart_type = spec.get("chart_type", "").lower()

    # Clean up null strings from LLM
    color = spec.get("color")
    if color == "null" or color is None:
        color = None
    orientation = spec.get("orientation", "v")

    builders = {
        "bar": lambda d, s: px.bar(
            d, x=s["x"], y=s["y"], color=color,
            title=s.get("title"), orientation=orientation, barmode="group",
        ),
        "line": lambda d, s: px.line(
            d, x=s["x"], y=s["y"], color=color, title=s.get("title"),
        ),
        "scatter": lambda d, s: px.scatter(
            d, x=s["x"], y=s["y"], color=color, title=s.get("title"),
        ),
        "pie": lambda d, s: px.pie(
            d, names=s["x"], values=s["y"], title=s.get("title"),
        ),
        "histogram": lambda d, s: px.histogram(
            d, x=s["x"], color=color, title=s.get("title"),
        ),
        "area": lambda d, s: px.area(
            d, x=s["x"], y=s["y"], color=color, title=s.get("title"),
        ),
    }

    builder = builders.get(chart_type)
    if not builder:
        logger.warning(f"[chart_generator] Unsupported chart type: {chart_type}")
        return None

    try:
        plot_df = df.copy()

        # Apply aggregation if specified
        agg_func = spec.get("agg_func")
        if agg_func and agg_func != "null" and chart_type != "histogram":
            group_cols = [spec["x"]]
            if color:
                group_cols.append(color)
            y_val = spec["y"]
            y_cols = y_val if isinstance(y_val, list) else [y_val]
            plot_df = plot_df.groupby(group_cols, as_index=False)[y_cols].agg(agg_func)

        fig = builder(plot_df, spec)
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            height=380,
            font=dict(
                family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                size=12,
            ),
        )
        return fig

    except Exception as e:
        logger.error(f"[chart_generator] Failed to build figure: {e}")
        return None


def generate_chart(
    df: pd.DataFrame,
    user_question: str,
    sql_description: Optional[str] = None,
    space_instructions: Optional[str] = None,
) -> Optional[go.Figure]:
    """
    Analyze a DataFrame and generate a plotly chart if appropriate.

    Calls an LLM to decide whether a chart is warranted and which type
    to use, then builds and returns a plotly Figure. Returns None if
    no chart is appropriate or if any step fails.

    Args:
        df: Query result data
        user_question: The user's original question
        sql_description: Genie's description of the SQL query
        space_instructions: Genie Space description/instructions for context

    Returns:
        plotly Figure or None
    """
    if df is None or df.empty or len(df) < 2:
        return None

    spec = _get_chart_spec(df, user_question, sql_description, space_instructions)
    if spec is None:
        return None

    return _build_figure(df, spec)
