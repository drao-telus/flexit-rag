"""
Image Analysis Module for RAG Validation

This module analyzes image information in RAG files and identifies missing images
for integration with the validation report system.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class ImageAnalysisModule:
    """Analyzes image information in RAG files for validation reports."""

    def __init__(self, process_images_dir: str = "crawler/process-images"):
        self.process_images_dir = Path(process_images_dir)
        self.image_index = self._build_image_index()

    def _build_image_index(self) -> Dict[str, Dict[str, str]]:
        """Build a comprehensive index of all images with their paths by category"""
        image_index = defaultdict(dict)

        if not self.process_images_dir.exists():
            return image_index

        # Scan all subdirectories for images
        for category_dir in self.process_images_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                for image_file in category_dir.glob("*.png"):
                    filename_without_ext = image_file.stem
                    relative_path = f"{category_name}/{image_file.name}"

                    # Store both with and without extension for flexible matching
                    image_index[category_name][image_file.name] = relative_path
                    image_index[category_name][filename_without_ext] = relative_path

        return dict(image_index)

    def _extract_image_references_from_content(self, content: str) -> List[str]:
        """Extract all image references from content"""
        patterns = [
            r"!\[([^\]]*)\]\(([^)]+)\)",  # Markdown image format ![alt](url)
            r"\[Image: [^-]+ - ([^\]]+)\]",  # [Image: description - filename]
        ]

        image_refs = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if pattern.startswith(r"!\["):
                # For markdown format, extract the URL and get filename
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        url = match[1]
                        # Extract filename from URL
                        if "/" in url:
                            filename = url.split("/")[-1]
                        else:
                            filename = url
                        image_refs.append(filename)
            else:
                # For other patterns, use the match directly
                if matches and isinstance(matches[0], tuple):
                    image_refs.extend(
                        [m[0] if isinstance(m, tuple) else m for m in matches]
                    )
                else:
                    image_refs.extend(matches)

        # Clean and deduplicate
        cleaned_refs = []
        for ref in image_refs:
            cleaned_ref = ref.strip()
            if cleaned_ref and cleaned_ref not in cleaned_refs:
                cleaned_refs.append(cleaned_ref)

        return cleaned_refs

    def _get_document_category(self, rag_filename: str) -> str:
        """Extract document category from RAG filename"""
        base_name = rag_filename.replace("_rag.json", "")

        category_patterns = [
            "Annual Enrollment Guide",
            "BeneficiaryMgmt",
            "Dataload",
            "EmailGuide",
            "Employee Admin Ref",
            "EmployeeSiteGuide",
            "HRIS",
            "OfflineTasks",
            "payroll",
            "Plan Setup Reference",
            "Template",
            "UploadFileGuide",
        ]

        for category in category_patterns:
            if base_name.startswith(category):
                return category

        return "Unknown"

    def _find_image_path(
        self, filename: str, document_category: str
    ) -> Optional[Tuple[str, str]]:
        """Find the best matching image path for a filename"""
        if not filename:
            return None

        # Check if category exists in image index
        if document_category not in self.image_index:
            return None

        category_images = self.image_index[document_category]

        # Try exact filename match first
        if filename in category_images:
            return category_images[filename], filename

        # Try with .png extension
        png_filename = f"{filename}.png"
        if png_filename in category_images:
            return category_images[png_filename], png_filename

        # Try without extension if it has one
        if "." in filename:
            base_filename = filename.split(".")[0]
            if base_filename in category_images:
                return category_images[base_filename], f"{base_filename}.png"

        return None

    def analyze_rag_file_images(
        self, rag_file_path: str, rag_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze image information for a single RAG file"""
        analysis = {
            "file_path": rag_file_path,
            "document_category": "",
            "images_in_array": [],
            "content_image_references": [],
            "missing_images": [],
            "mapped_images": [],
            "image_statistics": {
                "total_images_in_array": 0,
                "total_content_references": 0,
                "successfully_mapped": 0,
                "missing_count": 0,
                "mapping_success_rate": 0.0,
            },
        }

        try:
            # Extract document category
            filename = Path(rag_file_path).name
            analysis["document_category"] = self._get_document_category(filename)

            # Get existing images from array with latest structure
            existing_images = rag_json.get("images", [])
            analysis["images_in_array"] = [
                {
                    "image_id": img.get("image_id", ""),
                    "type": img.get("type", ""),
                    "filename": img.get("filename", ""),
                    "description": img.get("description", ""),
                    "local_path": img.get("local_path", ""),
                    "category": img.get("category", ""),
                    "exists": img.get("exists", False),
                    "image_url": img.get("image_url", ""),
                    "enhanced_image_url": img.get("enhanced_image_url", ""),
                    "original_src": img.get("original_src", ""),
                    "position_in_content": img.get("position_in_content", 0),
                }
                for img in existing_images
            ]
            analysis["image_statistics"]["total_images_in_array"] = len(existing_images)

            # Extract content from all chunks
            all_content = ""
            for chunk in rag_json.get("chunks", []):
                all_content += chunk.get("content", "") + " "

            # Find image references in content
            content_refs = self._extract_image_references_from_content(all_content)
            analysis["content_image_references"] = content_refs
            analysis["image_statistics"]["total_content_references"] = len(content_refs)

            # New approach: Check if images in the array are properly mapped
            # Since the RAG pipeline already includes images in the array, we should validate that
            for img in existing_images:
                img_filename = img.get("filename", "")
                img_exists = img.get("exists", False)

                if img_exists and img_filename:
                    analysis["mapped_images"].append(
                        {
                            "reference": img.get("description", ""),
                            "filename": img_filename,
                            "path": img.get("local_path", ""),
                            "status": "mapped",
                            "image_id": img.get("image_id", ""),
                            "type": img.get("type", ""),
                        }
                    )
                    analysis["image_statistics"]["successfully_mapped"] += 1
                else:
                    analysis["missing_images"].append(
                        {
                            "reference": img.get("description", ""),
                            "filename": img_filename,
                            "path": img.get("local_path", ""),
                            "status": "missing_file",
                            "reason": f"Image file does not exist: {img.get('local_path', 'Unknown path')}",
                            "image_id": img.get("image_id", ""),
                        }
                    )
                    analysis["image_statistics"]["missing_count"] += 1

            # Calculate mapping success rate based on images in array
            total_images = analysis["image_statistics"]["total_images_in_array"]
            if total_images > 0:
                success_rate = (
                    analysis["image_statistics"]["successfully_mapped"] / total_images
                ) * 100
                analysis["image_statistics"]["mapping_success_rate"] = round(
                    success_rate, 1
                )

        except Exception as e:
            analysis["error"] = str(e)

        return analysis

    def get_image_category_summary(self) -> Dict[str, int]:
        """Get summary of available images by category"""
        summary = {}
        for category, images in self.image_index.items():
            unique_images = len(set(images.values()))  # Count unique paths
            summary[category] = unique_images
        return summary

    def generate_image_analysis_summary(
        self, all_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate overall summary of image analysis across all files"""
        summary = {
            "total_files_analyzed": len(all_analyses),
            "files_with_images_in_array": 0,
            "files_with_content_references": 0,
            "files_with_missing_images": 0,
            "total_images_in_arrays": 0,
            "total_content_references": 0,
            "total_mapped_images": 0,
            "total_missing_images": 0,
            "overall_mapping_success_rate": 0.0,
            "category_breakdown": defaultdict(
                lambda: {"files": 0, "content_refs": 0, "mapped": 0, "missing": 0}
            ),
            "available_images_by_category": self.get_image_category_summary(),
        }

        for analysis in all_analyses:
            if "error" in analysis:
                continue

            stats = analysis["image_statistics"]
            category = analysis["document_category"]

            # File-level counts
            if stats["total_images_in_array"] > 0:
                summary["files_with_images_in_array"] += 1
            if stats["total_content_references"] > 0:
                summary["files_with_content_references"] += 1
            if stats["missing_count"] > 0:
                summary["files_with_missing_images"] += 1

            # Overall totals
            summary["total_images_in_arrays"] += stats["total_images_in_array"]
            summary["total_content_references"] += stats["total_content_references"]
            summary["total_mapped_images"] += stats["successfully_mapped"]
            summary["total_missing_images"] += stats["missing_count"]

            # Category breakdown
            summary["category_breakdown"][category]["files"] += 1
            summary["category_breakdown"][category]["content_refs"] += stats[
                "total_content_references"
            ]
            summary["category_breakdown"][category]["mapped"] += stats[
                "successfully_mapped"
            ]
            summary["category_breakdown"][category]["missing"] += stats["missing_count"]

        # Calculate overall success rate
        if summary["total_content_references"] > 0:
            overall_rate = (
                summary["total_mapped_images"] / summary["total_content_references"]
            ) * 100
            summary["overall_mapping_success_rate"] = round(overall_rate, 1)

        # Convert defaultdict to regular dict for JSON serialization
        summary["category_breakdown"] = dict(summary["category_breakdown"])

        return summary


def main():
    """Test the image analysis module"""
    analyzer = ImageAnalysisModule()

    # Test with a sample RAG file
    rag_output_dir = Path("crawler/result_data/rag_output")
    if rag_output_dir.exists():
        rag_files = list(rag_output_dir.glob("*_rag.json"))[
            :5
        ]  # Test with first 5 files

        all_analyses = []
        for rag_file in rag_files:
            try:
                with open(rag_file, "r", encoding="utf-8") as f:
                    rag_json = json.load(f)

                analysis = analyzer.analyze_rag_file_images(str(rag_file), rag_json)
                all_analyses.append(analysis)

                print(f"Analyzed {rag_file.name}:")
                print(f"  Category: {analysis['document_category']}")
                print(
                    f"  Images in array: {analysis['image_statistics']['total_images_in_array']}"
                )
                print(
                    f"  Content references: {analysis['image_statistics']['total_content_references']}"
                )
                print(
                    f"  Missing images: {analysis['image_statistics']['missing_count']}"
                )
                print(
                    f"  Success rate: {analysis['image_statistics']['mapping_success_rate']}%"
                )
                print()

            except Exception as e:
                print(f"Error analyzing {rag_file}: {e}")

        # Generate summary
        summary = analyzer.generate_image_analysis_summary(all_analyses)
        print("=== OVERALL SUMMARY ===")
        print(f"Total files analyzed: {summary['total_files_analyzed']}")
        print(
            f"Files with content references: {summary['files_with_content_references']}"
        )
        print(f"Files with missing images: {summary['files_with_missing_images']}")
        print(
            f"Overall mapping success rate: {summary['overall_mapping_success_rate']}%"
        )


if __name__ == "__main__":
    main()
