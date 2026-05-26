from typing import List, Optional
import streamlit as st
import pandas as pd
import json

from agent.state import UIComponent


def render_component(component: UIComponent):
    try:
        comp_type = component.get("type", "")
        title = component.get("title", "")
        payload = component.get("payload", {})

        if comp_type == "chart":
            chart_type = component.get("chart_type", "line")
            data = payload.get("data", [])
            x_col = payload.get("x_column", "")
            y_col = payload.get("y_column", "")

            if not data:
                st.info("No data available for chart")
                return

            df = pd.DataFrame(data)

            if title:
                st.subheader(title)

            if chart_type == "line":
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    st.line_chart(df, x=x_col, y=y_col)
                else:
                    st.line_chart(df)
            elif chart_type == "bar":
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    st.bar_chart(df, x=x_col, y=y_col)
                else:
                    st.bar_chart(df)
            elif chart_type == "area":
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    st.area_chart(df, x=x_col, y=y_col)
                else:
                    st.area_chart(df)
            elif chart_type == "scatter":
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    st.scatter_chart(df, x=x_col, y=y_col)
                else:
                    st.scatter_chart(df)
            elif chart_type == "pie":
                if y_col and y_col in df.columns:
                    val = df[y_col].sum()
                    st.metric(label=title or y_col, value=f"{val:,.1f}")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.dataframe(df, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)

        elif comp_type == "table":
            data = payload.get("data", [])
            if data:
                if title:
                    st.subheader(title)
                st.dataframe(pd.DataFrame(data), use_container_width=True)

        elif comp_type == "debug":
            st.write(payload)

        else:
            st.json(payload)
    except Exception as e:
        st.caption(f"Component render error: {e}")


def handle_agent_response(response: dict):
    final_text = response.get("final_response", "")
    ui_components = response.get("ui_components", [])

    if final_text:
        st.markdown(final_text)

    if ui_components:
        for component in ui_components:
            render_component(component)


def display_agent_metrics(response: dict, elapsed: float):
    tool_trace = response.get("tool_trace", [])
    tools_used = ", ".join(t.get("tool", "?") for t in tool_trace) if tool_trace else "none"
    docs_found = sum(t.get("doc_count", 0) for t in tool_trace)
    sql_rows = sum(t.get("row_count", 0) for t in tool_trace)
    chart_count = sum(t.get("payloads", 0) for t in tool_trace)

    metrics_parts = []
    if docs_found:
        metrics_parts.append(f"📄 {docs_found} docs")
    if sql_rows:
        metrics_parts.append(f"🗄️ {sql_rows} rows")
    if chart_count:
        metrics_parts.append(f"📊 {chart_count} charts")

    st.caption(
        f"🤖 Agent | Tools: {tools_used} | "
        f"{' | '.join(metrics_parts) if metrics_parts else ''} "
        f"| ⏱️ {elapsed:.2f}s"
    )
