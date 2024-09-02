# Enhanced ViLT Visual Question Answering System

This repository contains an interactive Gradio-based application that leverages the power of the ViLT (Vision-and-Language Transformer) model to perform Visual Question Answering (VQA). Users can upload an image and ask questions about it, and the model will provide answers along with confidence scores, alternative answers, and suggested follow-up questions based on the image content.

## Features

- **Visual Question Answering (VQA)**: Upload an image and ask a question about its content. The model will provide an answer based on the visual information.
- **Confidence Scores**: Get a confidence score for the provided answer to understand the model's certainty.
- **Alternative Answers**: View alternative answers that the model considered, along with their probabilities.
- **Suggested Questions**: Automatically generate relevant follow-up questions based on the content of the image.
- **Pre-trained Models**: Utilizes pre-trained ViLT and ResNet models for accurate and efficient image and text processing.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7+
- Gradio
- PyTorch
- Transformers
- torchvision
- Pillow (PIL)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/GirishWangikar/ViLT-Image-Q-A-System
    cd enhanced-vilt-vqa
    ```

2. Install the required packages:
    ```bash
    pip install gradio torch torchvision transformers Pillow
    ```

## Usage

1. Run the application:
    ```bash
    python app.py
    ```

2. Open your web browser and navigate to the URL provided in the console output.

3. In the Gradio interface:
    - Upload an image by clicking on the image input box.
    - Type a question related to the image in the text box.
    - Click the "Submit" button to receive the following outputs:
        - **Main Answer**: The primary answer to your question.
        - **Confidence Score**: The model's confidence in its main answer.
        - **Alternative Answers**: Other potential answers the model considered.
        - **Suggested Questions**: Additional questions you might want to ask based on the image content.

## Customization

- **Common Objects List**: Modify the `common_objects` list in the code to include different or additional categories for suggesting questions.
- **Model Parameters**: Adjust the top-k value in the `predict` function to change how many alternative answers are provided.

## Contact
Created by [Girish Wangikar](https://www.linkedin.com/in/girish-wangikar/)

Check out more on [LinkedIn](https://www.linkedin.com/in/girish-wangikar/) | [Portfolio](https://girishwangikar.github.io/Girish_Wangikar_Portfolio.github.io/)
