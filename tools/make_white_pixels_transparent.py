from PIL import Image
import os


def replace_non_black_with_transparent(image, threshold=10):
    img = image.convert("RGBA")
    data = img.getdata()
    new_data = []
    for item in data:
        if sum(item[:3]) > threshold:  # Check pixel darkness (sum of R, G, B values)
            new_data.append((255, 255, 255, 0))  # Replace with transparent pixel
        else:
            new_data.append(item)
    img.putdata(new_data)
    return img


def process_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):  # Assuming all images are PNG files
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            modified_image = replace_non_black_with_transparent(image)
            modified_image.save(image_path)
            print(f"Processed: {filename}")


if __name__ == "__main__":
    icons_folder = "ICONS"
    process_images(icons_folder)
    print("All images processed and saved.")
