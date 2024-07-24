"""
    Gradio Based UI for the Tomography Artifact Classification Inference
    This script is used to create a Gradio interface for the Tomography Artifact Classification Inference.
    Its uses apptainer.
"""

import gradio as gr
import torch
from torchvision import transforms
import os


labels = {"datasets-with-ring": 0, "datasets-no-ring": 1, "bad-center": 2}
labels_list = list(labels.keys())


def load_model(model_name):
    model_path = os.path.join("models", f"{model_name}.pth")
    model = torch.load(model_path)
    model.eval()
    return model


models = ["ViT", "CNN", "resnet50"]


def predict(inp, model_name, description):
    model = load_model(model_name)
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels_list[i]: float(prediction[i]) for i in range(len(labels))}
    return confidences, description


def predict(inp, model_name, description):
    model = load_model(model_name)
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels_list[i]: float(prediction[i]) for i in range(len(labels))}

    top_predictions = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]
    itemized_descriptions = []
    for i, (label, confidence) in enumerate(top_predictions):
        itemized_description = (
            f"Itemized Description {i+1}:\n"
            f"1. Model Name: {model_name}\n"
            f"2. Image Shape: {inp.shape}\n"
            f"3. Prediction: {label} ({confidence*100:.2f}%)\n"
            f"4. User Description: {description}"
        )
        itemized_descriptions.append(itemized_description)

    return confidences, "\n\n".join(itemized_descriptions)


gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil"),
        gr.Dropdown(choices=models, label="Model"),
        gr.Textbox(label="Description"),
    ],
    outputs=[gr.Label(num_top_classes=3), gr.Textbox(label="Itemized Descriptions")],
).launch()
