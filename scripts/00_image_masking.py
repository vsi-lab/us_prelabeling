import cv2
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent

input_root = project_root / "data" / "raw" / "images"
output_root = project_root / "data" / "raw" / "images_masked"


crop_top = 50
crop_left = 50


image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}



processed = 0
failed = 0

for image_path in input_root.rglob("*"):

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in image_extensions:
        continue

    relative_path = image_path.relative_to(input_root)

    output_path = output_root / relative_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"FAILED TO READ: {image_path}")
        failed += 1
        continue

    height, width = img.shape[:2]
    if crop_top >= height or crop_left >= width:
        print(f"CROP TOO LARGE: {image_path}")
        failed += 1
        continue
    cropped = img[crop_top:, crop_left:]

    success = cv2.imwrite(str(output_path), cropped)

    if success:
        processed += 1

        if processed % 500 == 0:
            print(f"Processed {processed} images...")
    else:
        print(f"FAILED TO SAVE: {output_path}")
        failed += 1



print("\nDone.")
print(f"Processed: {processed}")
print(f"Failed:    {failed}")
print(f"Saved to:  {output_root}")