from PIL import Image
import os

def convert_to_three_channels(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        image.save(image_path)

def traverse_and_convert(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                convert_to_three_channels(image_path)

root_dir = r'path'
traverse_and_convert(root_dir)
