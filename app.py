import gradio as gr
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import os
import shutil
import zipfile

# Load model and processor
model_name = "prithivMLmods/Deepfake-QualityAssess-88M"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# Create output directories
issue_dir = "Issue In Deepfake"
high_quality_dir = "High Quality Deepfake"
zip_filename = "classified_images.zip"
os.makedirs(issue_dir, exist_ok=True)
os.makedirs(high_quality_dir, exist_ok=True)

def deepfake_detection(images):
    """Predicts deepfake probability scores for multiple images, saves them in respective folders, and returns a zip file."""
    results = {}
    
    for image in images:
        image_name = os.path.basename(image.name)
        image = Image.open(image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        
        labels = model.config.id2label
        predictions = {labels[i]: round(probs[i], 3) for i in range(len(probs))}
        
        # Get the highest probability label
        max_label = max(predictions, key=predictions.get)
        max_score = predictions[max_label]
        
        # Save image in respective folder
        filename = f"{max_label}_{max_score:.3f}.png"
        save_dir = issue_dir if "issue" in max_label.lower() else high_quality_dir
        save_path = os.path.join(save_dir, filename)
        image.save(save_path)
        
        results[image_name] = {"predictions": predictions, "saved_as": filename}
    
    # Create a zip file
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for folder in [issue_dir, high_quality_dir]:
            for root, _, files in os.walk(folder):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file)))
    
    return results, zip_filename

# Create Gradio interface
iface = gr.Interface(
    fn=deepfake_detection,
    inputs=gr.File(file_types=["image"], file_count="multiple", label="Upload Images"),
    outputs=[gr.JSON(label="Prediction Scores"), gr.File(label="Download Classified Images (ZIP)")],
    title="Deepfake Quality Detection",
    description="Upload multiple images to check their deepfake probability scores. Images will be classified, stored, and available for download as a zip file."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()