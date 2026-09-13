from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print("Model berhasil dimuat!")
print("Label:", model.config.id2label)


def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

    label = model.config.id2label[predicted_class]

    return label, confidence


print("\n=== Indonesian Sentiment Analysis ===")
print("Ketik 'keluar' untuk mengakhiri program.\n")

while True:
    text = input("Masukkan kalimat: ")

    if text.lower() == "keluar":
        print("Program selesai.")
        break

    if not text.strip():
        print("Kalimat tidak boleh kosong.\n")
        continue

    label, confidence = predict_sentiment(text)

    print("Sentiment  :", label)
    print("Confidence :", f"{confidence:.2%}")
    print("-" * 50)