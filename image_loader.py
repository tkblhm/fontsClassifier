import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import os
import random

# Define the dataset directory
# DATASET_PATH = "./font_dataset"
DATASET_PATH = "D:/gyt/font_dataset"
FONT_CLASSES = ["ComicSansMS", "CourierNew", "Arial", "Helvetica", "Verdana", "Futura", "GillSans", "OpenSans", "Roboto", "Calibri", "TimesNewRoman", "Georgia", "Garamond", "Palatino", "Baskerville", "Consolas", "Tahoma", "BrushScript", "Impact", "CenturyGothic"]
FONT_PATH = "fonts"


# Ensure dataset directory exists
os.makedirs(DATASET_PATH, exist_ok=True)

# Parameters
NUM_IMAGES_PER_FONT = 400
IMAGE_SIZE = (64, 64)
TEXT_SAMPLES = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


# Function to generate an image with a given font
def generate_image(font_path, text, font_size=40, image_size=IMAGE_SIZE):
    img = Image.new("L", image_size, color=255)  # Create white background
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, size=font_size)  # Set font size
    except IOError:
        print(f"Font not found: {font_path}")
        return None
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]

    # text_width, text_height = draw.textsize(text, font=font)
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2
    draw.text((x, y), text, fill=0, font=font)  # Draw text in black

    return img

def generate_images():
    # Generate dataset
    for font in FONT_CLASSES:
        font_dir = os.path.join(DATASET_PATH, font)
        os.makedirs(font_dir, exist_ok=True)
        source_dir = os.path.join(FONT_PATH, font)
        for idx, path in enumerate(os.listdir(source_dir)):
            for i in range(NUM_IMAGES_PER_FONT):
                for size in range(14, 60, 2):
                    text = "".join(random.choices(TEXT_SAMPLES, k=random.randint(1, 6)))  # Random short text
                    img = generate_image(os.path.join(source_dir, path), text, font_size=size)

                    if img:
                        img.save(os.path.join(font_dir, f"{idx}-{size}-{i}.png"))

    print("Dataset generation complete.")

# Data Augmentation (Applied during loading)
transform = transforms.Compose([
    transforms.RandomRotation(10),       # Rotate image randomly by ±10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=(255,)),  # Small translation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
    transforms.ToTensor(),               # Convert image to tensor [0,1] range
    transforms.Normalize((0.5,), (0.5,)) # Normalize to mean 0, std 1
])

# Custom Dataset Class
class FontDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, font_name in enumerate(FONT_CLASSES):
            font_folder = os.path.join(root_dir, font_name)
            for img_name in os.listdir(font_folder):
                self.image_paths.append(os.path.join(font_folder, img_name))
                self.labels.append(label)  # ComicSansMS=0, CourierNew=1

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # Load image as grayscale
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        # return img, torch.tensor(label, dtype=torch.long)
        return img, torch.tensor(label)



if __name__ == '__main__':
    generate_images()
    # Load Dataset
    dataset = FontDataset(DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)