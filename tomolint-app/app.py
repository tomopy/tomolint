"""
    Gradio Based UI for the Tomography Artifact Classification Inference
    This script is used to create a Gradio interface for the Tomography Artifact Classification Inference.
    Its uses apptainer.
"""

import gradio as gr
import torch
from torchvision import transforms
import os
from tomolint.training import RingClassifier
import cv2


labels = {"datasets-with-ring": 0, "datasets-no-ring": 1, "bad-center": 2}
labels_list = list(labels.keys())


def load_model(model_name):
    model_path = os.path.join(
        "/Users/abayomi/Desktop/internship-24/tomolint/tomolint-app/models",
        f"{model_name}.ckpt",
    )
    hparams = {
        "vit_params": {
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "patch_size": 4,
            "num_channels": 1,
            "num_patches": 64,
            "num_classes": 3,
            "dropout": 0.2,
        },
        "optimizer_params": {
            "lr": 3e-4,
        },
    }
    model = RingClassifier(3, model_name, hparams)
    model = RingClassifier.load_from_checkpoint(model_path)

    model.eval()
    return model


models = ["vit", "cnn"]


def predict(inp, model_name, description):
    model = load_model(model_name)
    inp = transforms.Grayscale(num_output_channels=1)(inp)
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
