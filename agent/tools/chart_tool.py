from typing import Any
from langchain_core.tools import tool
from agent.config import agent_settings


def _serialize_val(val: Any) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


@tool
def generate_chart(chart_type: str, data: list, x_column: str, y_column: str, title: str = "") -> dict:
    """Generate a chart specification for data visualization.

    Use this tool when the user wants a visual representation of data.
    Supported chart types: line, bar, area, scatter, pie.

    Args:
        chart_type: Type of chart (line, bar, area, scatter, pie)
        data: List of dicts with x and y values
        x_column: Field name for x-axis
        y_column: Field name for y-axis
        title: Chart title (optional)
    """
    max_points = agent_settings.max_chart_data_points
    valid_types = {"line", "bar", "area", "scatter", "pie"}

    if chart_type not in valid_types:
        return {
            "component": "chart",
            "error": f"Unsupported chart type '{chart_type}'. Supported: {', '.join(sorted(valid_types))}",
        }

    if not data:
        return {
            "component": "chart",
            "error": "No data provided for chart",
        }

    truncated = data[:max_points]

    chart_payload = {
        "component": "chart",
        "chart_type": chart_type,
        "title": title or f"{y_column} by {x_column}",
        "payload": {
            "x_column": x_column,
            "y_column": y_column,
            "data": truncated,
        },
    }

    return chart_payload


@tool
def show_dataframe(data: list, title: str = "") -> dict:
    """Display data as a formatted table.

    Use this tool to present structured data in a table format
    when a chart would not be the best visualization.
    """
    if not data:
        return {"component": "table", "error": "No data provided"}

    return {
        "component": "table",
        "title": title or "Data Table",
        "payload": {
            "data": data[:500],
            "columns": list(data[0].keys()) if data else [],
        },
    }
