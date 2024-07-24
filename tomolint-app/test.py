
# import gradio as gr
# import torch
# import requests
# from torchvision import transforms

# # Function to load the model based on user selection
# def load_model(model_name):
#     return torch.hub.load("pytorch/vision:v0.6.0", model_name, pretrained=True).eval()

# # Available models
# models = ["resnet18", "resnet34", "resnet50"]

# # Download human-readable labels for ImageNet.
# response = requests.get("https://git.io/JJkYN")
# labels = response.text.split("\n")

# def predict(inp, model_name, description):
#     model = load_model(model_name)
#     inp = transforms.ToTensor()(inp).unsqueeze(0)
#     with torch.no_grad():
#         prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
#         confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
#     return confidences, description

# gr.Interface(
#     fn=predict,
#     inputs=[gr.Image(type="pil"), gr.Dropdown(choices=models, label="Model"), gr.Textbox(label="Description")],
#     outputs=[gr.Label(num_top_classes=3), gr.Textbox(label="Your Description")],
#     # examples=[["lion.jpg", "vision transformer", "This is a lion."], ["cheetah.jpg", "cnn ", "This is a cheetah."]],
# ).launch()



import gradio as gr
import torch
import requests
from torchvision import transforms

# Function to load the model based on user selection
def load_model(model_name):
    return torch.hub.load("pytorch/vision:v0.6.0", model_name, pretrained=True).eval()

# Available models
models = ["resnet18", "resnet34", "resnet50"]

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def predict(inp, model_name, description):
    model = load_model(model_name)
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
        
    # Create itemized descriptions for the top 3 predictions
    top_predictions = sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]
    itemized_descriptions = []
    for i, (label, confidence) in enumerate(top_predictions):
        itemized_description = (f"Itemized Description {i+1}:\n"
                                f"1. Model Name: {model_name}\n"
                                f"2. Image Shape: {inp.shape}\n"
                                f"3. Prediction: {label} ({confidence*100:.2f}%)\n"
                                f"4. User Description: {description}")
        itemized_descriptions.append(itemized_description)
    
    return confidences, "\n\n".join(itemized_descriptions)

gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Dropdown(choices=models, label="Model"), gr.Textbox(label="Description")],
    outputs=[gr.Label(num_top_classes=3), gr.Textbox(label="Itemized Descriptions")],
).launch()
