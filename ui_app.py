import json
import gradio as gr

from normalizer import normalize_claim, normalize_document_text, classify_document_type
from schema import ClaimAttributes

APP_TITLE = "Claims Description Normalizer"
APP_DESCRIPTION = (
    "Normalize insurance claim descriptions into reliable,"
    "structured JSON with a fully offline LLM pipeline"
)

# ---------- Core callback for free-text claims ----------

def run_normalizer_from_text(claim_text: str) -> tuple:
    claim_text = (claim_text or "").strip()
    if not claim_text:
        empty_json = json.dumps(ClaimAttributes().model_dump(), indent=2)
        return (
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "Low",
            "Please enter a claim description.",
            empty_json,
        )

    claim = normalize_claim(claim_text)
    data = claim.model_dump()

    # Human-readable summary fields
    loss_type = data["loss_type"]
    severity = data["severity"]
    asset = data["asset"]
    estimated_loss = data["estimated_loss"]
    incident_date = data["incident_date"]
    location = data["location"]
    confidence = data["confidence"]
    explanation = data["explanation"] or "Not provided"

    json_output = json.dumps(data, indent=2)

    return (
        loss_type,
        severity,
        asset,
        estimated_loss,
        incident_date,
        location,
        confidence,
        explanation,
        json_output,
    )

# ---------- (Next step) PDF helper we will hook up later ----------

def extract_text_from_pdf(pdf_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)

    return "\n".join(parts).strip()


def run_normalizer_from_pdf(pdf_file):
    if pdf_file is None:
        empty_json = json.dumps(ClaimAttributes().model_dump(), indent=2)
        return "", "No PDF uploaded.", empty_json

    raw_text = extract_text_from_pdf(pdf_file)
    if not raw_text:
        empty_json = json.dumps(ClaimAttributes().model_dump(), indent=2)
        return "", "Could not extract text from the PDF.", empty_json

    # 🔍 1) Classify the document first
    is_claim, doc_type, reason = classify_document_type(raw_text)

    if not is_claim:
        # Not a claim – show a clear message instead of fake JSON
        info = {
            "detected_type": doc_type,
            "message": "The uploaded PDF does not look like a claim description.",
            "reason": reason,
        }
        info_json = json.dumps(info, indent=2)
        return raw_text, info_json, info_json

    # ✅ 2) It is a claim document → normalize as before
    claim = normalize_document_text(raw_text)
    data = claim.model_dump()
    json_output = json.dumps(data, indent=2)

    return raw_text, json_output, json_output


# ---------- Build Gradio UI ----------

# Removed css=... (not supported in Gradio 6)
with gr.Blocks(title=APP_TITLE) as demo:
    # Inject the same CSS via a <style> tag in Markdown
    gr.Markdown(
        """
        <style>
        #app-container {max-width: 1200px; margin: 0 auto;}
        .card {border-radius: 8px; padding: 16px; background: #111827;}
        .badge {display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 0.8rem;}
        .badge-label {color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .04em;}
        </style>
        """
    )

    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(APP_DESCRIPTION)

    with gr.Tab("Normalize Claim"):
        with gr.Row():
            # LEFT: Input
            with gr.Column(scale=2):
                gr.Markdown("### Step 1 · Enter or select a claim")

                claim_text = gr.Textbox(
                    label="Claim description",
                    placeholder="Paste claim notes here...",
                    lines=8,
                )

                example = gr.Dropdown(
                    label="Sample claims",
                    choices=[
                        "Fire in living room due to short circuit, sofa and TV unit completely burnt.",
                        "Car accident on main road, rear bumper damaged and minor scratches on door. No injuries reported.",
                        "Water leakage from overhead tank damaging bedroom ceiling and walls.",
                    ],
                    value=None,
                )

                def use_sample(selected):
                    return selected or ""

                example.change(use_sample, inputs=example, outputs=claim_text)

                run_button = gr.Button("🔍 Normalize Claim", variant="primary")

            # RIGHT: Output
            with gr.Column(scale=3):
                gr.Markdown("### Step 2 · Review structured summary")

                with gr.Row():
                    loss_type = gr.Label(label="Loss Type")
                    severity = gr.Label(label="Severity")
                    confidence = gr.Label(label="Confidence")

                with gr.Row():
                    asset = gr.Label(label="Asset")
                    estimated_loss = gr.Label(label="Estimated Loss")

                with gr.Row():
                    incident_date = gr.Label(label="Incident Date")
                    location = gr.Label(label="Location")

                explanation = gr.Textbox(
                    label="Explanation",
                    lines=3,
                    interactive=False,
                )

                gr.Markdown("### Step 3 · JSON for system integration")
                json_output = gr.JSON(label="JSON output")

        run_button.click(
            fn=run_normalizer_from_text,
            inputs=claim_text,
            outputs=[
                loss_type,
                severity,
                asset,
                estimated_loss,
                incident_date,
                location,
                confidence,
                explanation,
                json_output,
            ],
        )

    # ======= NEXT STEP: PDF TAB =======

    with gr.Tab("Normalize from PDF"):
        gr.Markdown(
            "Upload a PDF containing claim details. "
            "The app will extract text and normalize it into structured JSON."
        )

        with gr.Row():
            with gr.Column(scale=2):
                pdf_file = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                run_pdf_button = gr.Button("📄 Extract & Normalize")

            with gr.Column(scale=3):
                extracted_text = gr.Textbox(
                    label="Extracted text from PDF",
                    lines=10,
                    interactive=False,
                )
                pdf_json_output = gr.JSON(
                    label="JSON output",
                )

        run_pdf_button.click(
            fn=run_normalizer_from_pdf,
            inputs=pdf_file,
            outputs=[extracted_text, pdf_json_output, pdf_json_output],
        )

if __name__ == "__main__":
    demo.launch()
