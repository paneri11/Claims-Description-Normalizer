import gradio as gr
from normalizer import normalize_claim, normalize_claim_as_json_str


SAMPLE_CLAIMS = [
    "Car accident on main road, rear bumper damaged and minor scratches on door. No injuries reported.",
    "Water leakage from bathroom pipeline seeped into the bedroom wall, paint and plaster damaged.",
    "Mobile phone slipped from hand, screen shattered and back panel cracked.",
    "Fire in living room due to short circuit, sofa and TV unit completely burnt.",
]


def normalize_for_ui(text: str):
    if not text or not text.strip():
        return (
            "Please enter a claim description.",
            {},
            ""
        )

    claim = normalize_claim(text)
    display_dict = claim.to_display_dict()
    json_str = normalize_claim_as_json_str(text)

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


with gr.Blocks(title="Claims Description Normalizer (Offline GenAI)") as demo:
    gr.Markdown(
        """
    # 🧾 Claims Description Normalizer

    Convert messy insurance claim notes into clean, structured JSON using an **offline Mistral LLM (llama.cpp)**.
    This demo extracts key attributes like loss type, severity, and affected asset for downstream processing.
    """
    )

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

        with gr.Column(scale=3):
            summary_output = gr.Markdown(label="Summarized view")
            dict_output = gr.JSON(label="Structured attributes (dict)")
            json_output = gr.Textbox(
                label="JSON output",
                lines=10,
                show_copy_button=True,
            )

    def fill_example(example_text):
        return example_text

    use_example_btn.click(
        fn=fill_example,
        inputs=[examples],
        outputs=[claim_input],
    )

    run_btn = gr.Button("🔍 Normalize Claim")

    run_btn.click(
        fn=normalize_for_ui,
        inputs=[claim_input],
        outputs=[summary_output, dict_output, json_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
