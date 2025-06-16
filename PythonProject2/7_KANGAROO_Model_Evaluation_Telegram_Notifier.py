import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import cv2
import numpy as np
import asyncio

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition ---
class IMAGE_CLASSIFICATION(nn.Module):
    def __init__(self, num_classes=2):
        super(IMAGE_CLASSIFICATION, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# --- Load model ---
model = IMAGE_CLASSIFICATION(num_classes=2).to(device)
MODEL_PATH = r"D:\KANGROO\IMAGE_CLASSIFICATION_CYNDRELLA_8747.pth"  # Update your path
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Image transforms ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Predict image ---
def predict_image(img: Image.Image) -> str:
    image = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    classes = ["Fake", "Real"]
    return f"🖼️ Image Prediction: {classes[pred.item()]} (Confidence: {conf.item() * 100:.1f}%)"

# --- Predict video ---
async def predict_video(video_path: str, update: Update, context: ContextTypes.DEFAULT_TYPE, frame_sample_rate=5):
    cap = cv2.VideoCapture(video_path)
    frame_preds = []
    frame_confs = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = 0

    msg = await update.message.reply_text("⏳ Analyzing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if processed % frame_sample_rate == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            image = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                probs = F.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                frame_preds.append(pred.item())
                frame_confs.append(conf.item())

        processed += 1

        if processed % 30 == 0:
            percent = int((processed / total_frames) * 100)
            await msg.edit_text(f"📹 Video Processing: {min(percent, 100)}%")

    cap.release()

    if not frame_preds:
        await msg.edit_text("⚠️ Could not extract enough frames.")
        return

    pred_class = max(set(frame_preds), key=frame_preds.count)
    confs_for_class = [conf for p, conf in zip(frame_preds, frame_confs) if p == pred_class]
    avg_conf = sum(confs_for_class) / len(confs_for_class)

    classes = ["Fake", "Real"]
    await msg.edit_text(f"✅ Video Prediction: {classes[pred_class]} (Confidence: {avg_conf * 100:.1f}%)")

# --- Telegram handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! Upload an image or video and I’ll predict whether it’s Real or Fake with a confidence score."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await update.message.photo[-1].get_file()
        img_bytes = await file.download_as_bytearray()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        msg = await update.message.reply_text("⏳ Processing image...")
        result = predict_image(image)
        await msg.edit_text(result)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error processing image: {e}")

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        file = await update.message.video.get_file()
        temp_path = "temp_video.mp4"
        await file.download_to_drive(temp_path)
        await predict_video(temp_path, update, context)
        os.remove(temp_path)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error processing video: {e}")

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❗ I can only process images or videos for prediction.")

# --- Main ---
def main():
    TOKEN = "7892997172:AAGhEOHQLqKAxQfjlctnrBblwsrDSsU377o"
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.COMMAND, unknown))

    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()
