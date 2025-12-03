import os
from datetime import datetime

import gradio as gr
import pandas as pd
import plotly.express as px

from normalizer import normalize_claim, normalize_claim_as_json_str

# Path to store history
DATA_DIR = "data"
HISTORY_PATH = os.path.join(DATA_DIR, "claims_history.csv")

os.makedirs(DATA_DIR, exist_ok=True)


SAMPLE_CLAIMS = [
    "Car accident on main road, rear bumper damaged and minor scratches on door. No injuries reported.",
    "Water leakage from bathroom pipeline seeped into the bedroom wall, paint and plaster damaged.",
    "Mobile phone slipped from hand, screen shattered and back panel cracked.",
    "Fire in living room due to short circuit, sofa and TV unit completely burnt.",
]


def append_to_history(claim_text: str, record: dict):
    """
    Append a single normalized claim record to the CSV history.
    """
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "claim_text": claim_text,
        "loss_type": record.get("loss_type", "Unknown"),
        "severity": record.get("severity", "Unknown"),
        "asset": record.get("asset", "Unknown"),
        "estimated_loss": record.get("estimated_loss", "Unknown"),
        "incident_date": record.get("incident_date", "Unknown"),
        "location": record.get("location", "Unknown"),
        "confidence": record.get("confidence", "Unknown"),
        "explanation": record.get("explanation", "Not provided"),
    }

    df_new = pd.DataFrame([row])

    if os.path.exists(HISTORY_PATH):
        df_new.to_csv(HISTORY_PATH, mode="a", index=False, header=False)
    else:
        df_new.to_csv(HISTORY_PATH, mode="w", index=False, header=True)


def load_history():
    """
    Load history CSV if it exists. Return an empty DataFrame otherwise.
    """
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "claim_text",
                "loss_type",
                "severity",
                "asset",
                "estimated_loss",
                "incident_date",
                "location",
                "confidence",
                "explanation",
            ]
        )
    return pd.read_csv(HISTORY_PATH)


def build_analytics(df: pd.DataFrame):
    """
    Build Plotly charts for severity distribution and loss type distribution.
    """
    if df.empty:
        return None, None

    # Count by severity
    sev_counts = (
        df.groupby("severity")["timestamp"]
        .count()
        .reset_index()
        .rename(columns={"timestamp": "count"})
    )
    fig_severity = px.bar(
        sev_counts,
        x="severity",
        y="count",
        title="Claims by Severity",
        text="count",
    )

    # Count by loss_type
    lt_counts = (
        df.groupby("loss_type")["timestamp"]
        .count()
        .reset_index()
        .rename(columns={"timestamp": "count"})
    )
    fig_loss_type = px.bar(
        lt_counts,
        x="loss_type",
        y="count",
        title="Claims by Loss Type",
        text="count",
    )

    return fig_severity, fig_loss_type


def normalize_for_ui(text: str):
    if not text or not text.strip():
        return (
            "Please enter a claim description.",
            {},
            "",
        )

    claim = normalize_claim(text)
    display_dict = claim.to_display_dict()
    json_str = normalize_claim_as_json_str(text)

    # Save to history
    append_to_history(text, display_dict)

    # Short human-readable summary
    summary_lines = [
        f"**Loss Type:** {display_dict['loss_type']}",
        f"**Severity:** {display_dict['severity']}",
        f"**Asset:** {display_dict['asset']}",
        f"**Estimated Loss:** {display_dict['estimated_loss']}",
        f"**Location:** {display_dict['location']}",
        f"**Confidence:** {display_dict['confidence']}",
        "",
        f"**Explanation:** {display_dict['explanation']}",
    ]
    summary_md = "\n".join(summary_lines)
    return summary_md, display_dict, json_str


def refresh_history_and_charts():
    """
    Gradio callback: load history and build analytics.
    """
    df = load_history()
    fig_severity, fig_loss_type = build_analytics(df)
    return df, fig_severity, fig_loss_type


with gr.Blocks(title="Claims Description Normalizer (Offline GenAI)") as demo:
    gr.Markdown(
        """
    # 🧾 Claims Description Normalizer

    Convert messy insurance claim notes into clean, structured JSON using an **offline Mistral LLM (llama.cpp)**.
    This demo extracts key attributes like loss type, severity, and affected asset for downstream processing.
    """
    )

    with gr.Tabs():
        # ----- Tab 1: Normalize Claim -----
        with gr.Tab("Normalize Claim"):
            with gr.Row():
                with gr.Column(scale=2):
                    claim_input = gr.Textbox(
                        label="Enter claim description",
                        placeholder="Paste or type raw claim notes from adjusters or customers...",
                        lines=7,
                    )
                    examples = gr.Dropdown(
                        choices=SAMPLE_CLAIMS,
                        label="Sample claims",
                        info="Select a sample claim to quickly test the model.",
                    )
                    use_example_btn = gr.Button("Use selected sample")
                    run_btn = gr.Button("🔍 Normalize Claim")

                with gr.Column(scale=3):
                    summary_output = gr.Markdown(label="Summarized view")
                    dict_output = gr.JSON(label="Structured attributes (dict)")
                    json_output = gr.Textbox(
                        label="JSON output",
                        lines=10,
                    )

            def fill_example(example_text):
                return example_text

            use_example_btn.click(
                fn=fill_example,
                inputs=[examples],
                outputs=[claim_input],
            )

            run_btn.click(
                fn=normalize_for_ui,
                inputs=[claim_input],
                outputs=[summary_output, dict_output, json_output],
            )

        # ----- Tab 2: History & Analytics -----
        with gr.Tab("History & Analytics"):
            gr.Markdown(
                """
                ### 📊 Claim History & Basic Analytics

                Review previously processed claims and see simple trends by severity and loss type.
                """
            )
            refresh_btn = gr.Button("🔄 Refresh history and charts")

            history_table = gr.Dataframe(
                label="Processed Claims History",
                interactive=False,
                wrap=True,
            )

            severity_plot = gr.Plot(label="Claims by Severity")
            loss_type_plot = gr.Plot(label="Claims by Loss Type")

            # When refresh is clicked, load history and build charts
            refresh_btn.click(
                fn=refresh_history_and_charts,
                inputs=[],
                outputs=[history_table, severity_plot, loss_type_plot],
            )

if __name__ == "__main__":
    demo.launch()
