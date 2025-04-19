import streamlit as st
from PIL import Image
import random
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO


model = YOLO('masknet.pt')  # Load the pre-trained model
# Replace with your model and image directory paths
def predict_and_plot(model, image_path):
  results = model.predict(source=image_path)
  img = Image.open(image_path)
  fig, ax = plt.subplots(figsize=(10, 10))
  ax.imshow(img)
  ax.axis('off')

  for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cls = int(box.cls[0])
    confidence = box.conf[0].item()

    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=2, fill=False))
    label = f"{model.names[cls]}: {confidence:.2f}"
    ax.text(x1, y1, label, color='white', fontsize=12, backgroundcolor='red')

  return fig

def main():
  """Streamlit App for Mask Detection"""

  # Title and description
  st.title("Mask Detection App")

  # Display random predictions section
  st.header("Random Image Predictions")
  num_images = st.slider("Number of Random Images:", 1, 5, 3)  # Adjust slider range as needed
  images_dir = "images"  # Replace with your directory path

  if images_dir:
    image_files = random.sample([f for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.jpg')], num_images)
    cols = st.columns(num_images)
    for i, image_file in enumerate(image_files):
      image_path = os.path.join(images_dir, image_file)
      fig = predict_and_plot(model, image_path)
      cols[i].pyplot(fig)
  else:
    st.write("Please specify a directory containing images for random predictions.")


if __name__ == '__main__':
  main()