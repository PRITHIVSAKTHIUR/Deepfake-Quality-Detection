https://github.com/user-attachments/assets/8069bed1-4591-4898-ae71-69e59cc05633

# Deepfake Quality Detection

This repository provides a Gradio-based web interface for **Deepfake Quality Assessment** using a ViT (Vision Transformer) model. It classifies uploaded images into different quality categories, stores them in corresponding folders, and allows users to download the sorted results in a ZIP file.

## Demo

Upload multiple image files to the app, and it will:
- Predict deepfake probability scores
- Classify them as either:
  - **Issue In Deepfake**  
  - **High Quality Deepfake**
- Save them to organized folders
- Return a downloadable ZIP archive of the sorted images

##  Model Used

- **Model Name:** `prithivMLmods/Deepfake-QualityAssess-88M`  
- **Architecture:** Vision Transformer (ViT)  
- **Hosted on:** 🤗 [Hugging Face Hub](https://huggingface.co/prithivMLmods/Deepfake-QualityAssess-88M)

## Interface Preview

Powered by **Gradio**, this app provides a simple drag-and-drop UI:

![App Screenshot](https://user-images.githubusercontent.com/yourusername/your-screenshot.png) <!-- Replace with actual screenshot -->

##  Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Deepfake-Quality-Detection.git
cd Deepfake-Quality-Detection
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the dependencies**

```bash
pip install -r requirements.txt
```

> Example `requirements.txt`:
```txt
torch
transformers
gradio
Pillow
```

4. **Run the app**

```bash
python app.py
```

This will launch the Gradio interface in your browser.

## Output Structure

When you upload images:
- They are stored and sorted into:
  - `Issue In Deepfake/`
  - `High Quality Deepfake/`
- A ZIP file `classified_images.zip` is generated containing all classified images.

## 📄 Example Output

```json
{
  "image1.jpg": {
    "predictions": {
      "High Quality Deepfake": 0.912,
      "Issue In Deepfake": 0.088
    },
    "saved_as": "High Quality Deepfake_0.912.png"
  }
}
```

##  How It Works

- Loads a pretrained Vision Transformer (ViT) model.
- Processes each image and classifies based on softmax probability scores.
- Assigns to folders based on the label with the highest score.

## Credits

- Hugging Face Transformers
- Gradio Team
- Model by [@prithivMLmods](https://huggingface.co/prithivMLmods)

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.
