import gradio as gr
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch.nn.functional as F
from torchvision.models import resnet50
import torchvision.transforms as transforms

# Load the ViLT model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load pre-trained ResNet model
resnet50_model = resnet50(pretrained=True)
resnet50_model.eval()

# Simplified list of common objects
common_objects = ['person', 'animal', 'vehicle', 'furniture', 'electronic device', 'food', 'plant', 'building', 'clothing', 'sports equipment']

def get_image_features(image, model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features

def suggest_questions(image):
    features = get_image_features(image, resnet50_model)
    _, predicted = features.max(1)
    class_name = common_objects[predicted.item() % len(common_objects)]
    
    suggested_questions = [
        f"What is the main object in this image?",
        f"Is there a {class_name} in this picture?",
        "What colors are prominent in this image?",
        "What is the setting or background of this image?",
        "Are there any people in this image?"
    ]
    return suggested_questions

def predict(image, question):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    encoding = processor(image, question, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
    
    # Get top 5 answers and their probabilities
    top_5_probs, top_5_indices = probs.topk(5)
    
    answers = []
    for prob, idx in zip(top_5_probs[0], top_5_indices[0]):
        answer = model.config.id2label[idx.item()]
        answers.append((answer, prob.item()))
    
    main_answer = answers[0][0]
    confidence = answers[0][1]
    
    alternative_answers = [f"{ans} ({prob:.2f})" for ans, prob in answers[1:]]
    
    suggested_questions = suggest_questions(image)
    
    return (
        main_answer,
        f"{confidence:.2f}",
        ", ".join(alternative_answers),
        "\n".join(suggested_questions)
    )

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy"),
        gr.Textbox(lines=1, placeholder="Ask a question...")
    ],
    outputs=[
        gr.Textbox(label="Main Answer"),
        gr.Textbox(label="Confidence Score"),
        gr.Textbox(label="Alternative Answers"),
        gr.Textbox(label="Suggested Questions")
    ],
    title="Enhanced ViLT Visual Question Answering",
    description="Upload an image and ask a question about it. The model will provide the main answer, confidence score, alternative answers, and suggest additional questions."
)

# Launch the Gradio interface
interface.launch()
