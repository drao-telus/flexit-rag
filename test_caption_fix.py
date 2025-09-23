#!/usr/bin/env python3
"""Test script to verify caption fix."""

import json
from pathlib import Path
import sys

sys.path.append("crawler")

from md_pipeline.md_pipeline_main import MDPipelineOrchestrator, PipelineConfig


def test_caption_fix():
    """Test if captions are properly used as image descriptions."""

    # Test with the Annual Enrollment Guide file
    html_file = Path(
        "crawler/result_data/filtered_content/Annual Enrollment Guide_Add the next year in Plan Setup (Add Year).html"
    )
    if not html_file.exists():
        print("HTML file not found")
        return

    config = PipelineConfig(
        input_dir="crawler/result_data/filtered_content",
        output_dir="crawler/result_data/rag_output",
    )
    pipeline = MDPipelineOrchestrator(config)
    result = pipeline.process_single_file(str(html_file))
    print("Processing completed successfully")

    # Check the output file
    output_file = Path(
        "crawler/result_data/rag_output/Annual Enrollment Guide_Add the next year in Plan Setup (Add Year)_rag.json"
    )
    if not output_file.exists():
        print("Output file not found")
        return

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f'Found {len(data.get("images_metadata", []))} images')

    # Check first few images for caption usage
    for i, img in enumerate(data.get("images_metadata", [])[:3]):
        print(f"Image {i+1}:")
        print(f'  Description: {img.get("description", "")}')
        print(f'  Type: {img.get("type", "")}')
        print(f'  Enhanced URL: {img.get("enhanced_image_url", "")}')
        print()

    # Check if 'Benefit Plan setup page' appears in markdown content
    markdown_content = data.get("markdown_content", "")
    if "Benefit Plan setup page" in markdown_content:
        print("WARNING: Caption still appears in markdown content")
        # Find the context around it
        lines = markdown_content.split("\n")
        for i, line in enumerate(lines):
            if "Benefit Plan setup page" in line:
                print(f"Line {i+1}: {line}")
                if i > 0:
                    print(f"Previous line: {lines[i-1]}")
                if i < len(lines) - 1:
                    print(f"Next line: {lines[i+1]}")
                break
    else:
        print("SUCCESS: Caption no longer appears in markdown content")

    # Look for images with "Benefit Plan setup page" as description
    caption_found_in_description = False
    for img in data.get("images_metadata", []):
        if img.get("description") == "Benefit Plan setup page":
            caption_found_in_description = True
            print(
                f'SUCCESS: Found image with caption as description: {img.get("description")}'
            )
            break

    if not caption_found_in_description:
        print("WARNING: Caption not found in any image description")


if __name__ == "__main__":
    test_caption_fix()
