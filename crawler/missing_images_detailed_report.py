#!/usr/bin/env python3
"""
Generate detailed report of missing images with the files that reference them
"""

import sys
import re
from pathlib import Path
from bs4 import BeautifulSoup
from collections import defaultdict, Counter

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_missing_images_report():
    """Generate detailed report of missing images and the files that reference them."""

    filtered_content_dir = Path("crawler/result_data/filtered_content")
    process_images_dir = Path("crawler/process-images")

    # Get all available image files
    available_images = set()
    for img_file in process_images_dir.rglob("*"):
        if img_file.is_file() and img_file.suffix.lower() in [
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".webp",
        ]:
            available_images.add(img_file.name.lower())

    # Process HTML files
    html_files = list(filtered_content_dir.glob("*.html"))

    missing_images_by_file = defaultdict(list)  # file -> list of missing images
    missing_images_by_image = defaultdict(list)  # image -> list of files
    all_missing_refs = []

    for html_file in html_files:
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "html.parser")
            img_tags = soup.find_all("img")

            file_missing_images = []

            for img in img_tags:
                src = img.get("src", "").strip()
                if src:
                    filename = Path(src).name
                    base_filename = filename
                    if "_thumb_" in filename:
                        base_filename = re.sub(r"_thumb_\d+_\d+", "", filename)

                    # Check if image exists
                    found = False
                    for check_name in [filename.lower(), base_filename.lower()]:
                        if check_name in available_images:
                            found = True
                            break

                    if not found:
                        file_missing_images.append(filename)
                        missing_images_by_image[filename].append(html_file.name)
                        all_missing_refs.append(filename)

            if file_missing_images:
                missing_images_by_file[html_file.name] = file_missing_images

        except Exception as e:
            print(f"Error processing {html_file.name}: {e}")

    # Generate report
    print("=" * 80)
    print("DETAILED MISSING IMAGES REPORT")
    print("=" * 80)
    print(f"Total HTML files: {len(html_files)}")
    print(f"Files with missing images: {len(missing_images_by_file)}")
    print(f"Total missing image references: {len(all_missing_refs)}")
    print(f"Unique missing images: {len(missing_images_by_image)}")

    print("\n" + "=" * 80)
    print("MISSING IMAGES BY FILE")
    print("=" * 80)

    for file_name, missing_images in sorted(missing_images_by_file.items()):
        print(f"\n📁 {file_name}")
        print(f"   Missing images: {len(missing_images)}")
        for img in missing_images:
            print(f"     - {img}")

    print("\n" + "=" * 80)
    print("MISSING IMAGES BY IMAGE (with referencing files)")
    print("=" * 80)

    missing_counter = Counter(all_missing_refs)
    for i, (missing_img, count) in enumerate(missing_counter.most_common(), 1):
        print(f"\n{i:2d}. {missing_img}")
        print(f"    Referenced {count} times in files:")
        for file_name in sorted(missing_images_by_image[missing_img]):
            print(f"      - {file_name}")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Available images in folder: {len(available_images)}")
    print(f"Files with missing images: {len(missing_images_by_file)}")
    print(f"Unique missing images: {len(missing_images_by_image)}")
    print(f"Total missing references: {len(all_missing_refs)}")

    # Files with most missing images
    files_by_missing_count = [
        (len(missing), file_name)
        for file_name, missing in missing_images_by_file.items()
    ]
    files_by_missing_count.sort(reverse=True)

    print(f"\nTOP 10 FILES WITH MOST MISSING IMAGES:")
    for i, (missing_count, file_name) in enumerate(files_by_missing_count[:10], 1):
        print(f"{i:2d}. {file_name} ({missing_count} missing images)")

    return {
        "missing_images_by_file": dict(missing_images_by_file),
        "missing_images_by_image": dict(missing_images_by_image),
        "total_missing_refs": len(all_missing_refs),
        "unique_missing_images": len(missing_images_by_image),
        "files_with_missing_images": len(missing_images_by_file),
    }


if __name__ == "__main__":
    result = generate_missing_images_report()
