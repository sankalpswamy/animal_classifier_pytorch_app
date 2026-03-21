from __future__ import annotations

import gradio as gr
from PIL import Image

from bios import DEFAULT_BIOS
from inference import AnimalPredictor

TITLE = "Animal Image Classifier"
DESCRIPTION = (
    "Upload a photo of one of the 15 selected animals and the app will show the top predictions, confidence scores, "
    "and a short bio for the best match."
)

CSS = """
.gradio-container {max-width: 1150px !important;}
.hero h1, .hero p {text-align: center;}
"""

try:
    predictor = AnimalPredictor()
    metrics = predictor.load_metrics()
    startup_error = ""
except Exception as exc:
    predictor = None
    metrics = {}
    startup_error = str(exc)


def classify_animal(image: Image.Image):
    if predictor is None:
        return {}, "Model not ready", f"Train the model first. {startup_error}", ""
    if image is None:
        return {}, "No image uploaded", "Please upload an image.", ""

    predictions = predictor.predict(image, top_k=3)
    scores = {label: round(prob, 4) for label, prob in predictions}
    top_label, top_score = predictions[0]
    summary = f"Top prediction: {top_label} ({top_score * 100:.2f}%)"
    bio = DEFAULT_BIOS.get(top_label.lower(), f"No bio available yet for {top_label}.")
    return scores, summary, bio, top_label


with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
    with gr.Column(elem_classes="hero"):
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload an animal image")
            submit = gr.Button("Predict", variant="primary")
            clear = gr.Button("Clear")
        with gr.Column(scale=1):
            probs_output = gr.Label(label="Top predictions")
            summary_output = gr.Textbox(label="Prediction summary")
            bio_output = gr.Textbox(label="Animal bio", lines=4)
            class_output = gr.Textbox(label="Best class")

    with gr.Accordion("Latest training results", open=False):
        gr.JSON(value={
            "best_val_accuracy": metrics.get("val_accuracy"),
            "best_test_accuracy": metrics.get("test_accuracy"),
            "best_generated_test_accuracy": metrics.get("generated_test_accuracy"),
            "classes": metrics.get("classes", []),
        })

    submit.click(classify_animal, inputs=image_input, outputs=[probs_output, summary_output, bio_output, class_output])
    clear.click(lambda: (None, {}, "", "", ""), outputs=[image_input, probs_output, summary_output, bio_output, class_output])


if __name__ == "__main__":
    demo.launch()
