from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model architecture
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Load checkpoint
MODEL_PATH = "bert_text_classification_model.pt"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

# Set model to evaluation mode
model.eval()

def classify_text(text):
    """Classify input text as Normal (0) or AI-Generated (1)."""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    return "Normal (0)" if predicted_label == 0 else "AI-Generated (1)"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        prediction = classify_text(text)
        return jsonify({"result": prediction})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
