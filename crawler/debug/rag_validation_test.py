"""
RAG Validation Test

This script validates that the RAG pipeline captures all important content from HTML files
by comparing the original HTML content with the generated RAG output.
"""

import os
import json
import random
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from text_extractor import TextExtractor


class RAGValidationTest:
    """Main class for RAG validation testing."""

    def __init__(self, base_path: str = "crawler/result_data"):
        self.base_path = Path(base_path)
        self.filtered_content_path = self.base_path / "filtered_content"
        self.rag_output_path = self.base_path / "rag_output"
        self.report_path = Path("crawler/debug/report")
        self.text_extractor = TextExtractor()

        # Ensure report directory exists
        self.report_path.mkdir(parents=True, exist_ok=True)

    def get_html_files(self) -> List[Path]:
        """Get all HTML files from filtered_content directory."""
        try:
            html_files = list(self.filtered_content_path.glob("*.html"))
            print(f"Found {len(html_files)} HTML files in filtered_content")
            return html_files
        except Exception as e:
            print(f"Error reading HTML files: {e}")
            return []

    def select_random_files(
        self, html_files: List[Path], count: int = 10
    ) -> List[Path]:
        """Randomly select specified number of HTML files for testing."""
        if len(html_files) <= count:
            print(f"Using all {len(html_files)} available files (requested {count})")
            return html_files

        selected_files = random.sample(html_files, count)
        print(f"Randomly selected {len(selected_files)} files for validation")
        return selected_files

    def select_specific_file(self, html_files: List[Path], filename: str) -> List[Path]:
        """Select a specific HTML file for testing."""
        # Try exact match first
        for html_file in html_files:
            if html_file.name == filename:
                print(f"Found exact match: {filename}")
                return [html_file]

        # Try partial match
        matching_files = [f for f in html_files if filename.lower() in f.name.lower()]
        if matching_files:
            selected_file = matching_files[0]
            print(f"Found partial match: {selected_file.name} for '{filename}'")
            return [selected_file]

        print(f"No file found matching: {filename}")
        return []

    def select_multiple_files(
        self, html_files: List[Path], filenames: List[str]
    ) -> List[Path]:
        """Select multiple specific HTML files for testing."""
        selected_files = []
        not_found_files = []

        for filename in filenames:
            # Try exact match first
            found = False
            for html_file in html_files:
                if html_file.name == filename:
                    selected_files.append(html_file)
                    print(f"✓ Found exact match: {filename}")
                    found = True
                    break

            if not found:
                # Try partial match
                matching_files = [
                    f for f in html_files if filename.lower() in f.name.lower()
                ]
                if matching_files:
                    selected_file = matching_files[0]
                    selected_files.append(selected_file)
                    print(
                        f"✓ Found partial match: {selected_file.name} for '{filename}'"
                    )
                else:
                    not_found_files.append(filename)
                    print(f"✗ No file found matching: {filename}")

        if not_found_files:
            print(f"\nWarning: {len(not_found_files)} files not found:")
            for file in not_found_files:
                print(f"  - {file}")

        print(f"\nSelected {len(selected_files)} files for validation")
        return selected_files

    def find_corresponding_rag_file(self, html_file: Path) -> Path:
        """Find the corresponding RAG JSON file for an HTML file."""
        # Convert HTML filename to RAG filename
        html_name = html_file.stem  # filename without extension
        rag_filename = f"{html_name}_rag.json"
        rag_file_path = self.rag_output_path / rag_filename

        return rag_file_path

    def validate_file_pair(self, html_file: Path, rag_file: Path) -> Dict[str, Any]:
        """Validate a single HTML-RAG file pair."""
        result = {
            "html_file": str(html_file.name),
            "rag_file": str(rag_file.name),
            "html_exists": html_file.exists(),
            "rag_exists": rag_file.exists(),
            "validation_successful": False,
            "error": None,
            "comparison_results": None,
        }

        try:
            if not html_file.exists():
                result["error"] = f"HTML file not found: {html_file}"
                return result

            if not rag_file.exists():
                result["error"] = f"RAG file not found: {rag_file}"
                return result

            # Read HTML content
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Read RAG JSON content
            with open(rag_file, "r", encoding="utf-8") as f:
                rag_json_content = json.load(f)

            # Extract text from both sources
            html_text = self.text_extractor.extract_html_text(html_content)
            rag_text = self.text_extractor.extract_rag_text(rag_json_content)

            # Compare texts
            comparison_results = self.text_extractor.compare_texts(html_text, rag_text)

            # Add actual content for HTML report
            comparison_results["html_text"] = html_text
            comparison_results["rag_text"] = rag_text
            comparison_results["rag_json"] = rag_json_content

            result["comparison_results"] = comparison_results
            result["validation_successful"] = True

            print(
                f"✓ Validated {html_file.name} - Coverage: {comparison_results['coverage_percentage']}%"
            )

        except Exception as e:
            result["error"] = str(e)
            print(f"✗ Error validating {html_file.name}: {e}")

        return result

    def run_validation(
        self,
        file_count: int = 10,
        specific_file: str = None,
        specific_files: List[str] = None,
    ) -> Dict[str, Any]:
        """Run the complete validation process."""
        print("=" * 60)
        print("RAG VALIDATION TEST")
        print("=" * 60)

        # Get HTML files
        html_files = self.get_html_files()
        if not html_files:
            return {"error": "No HTML files found in filtered_content directory"}

        # Select files based on parameters
        if specific_files:
            selected_files = self.select_multiple_files(html_files, specific_files)
            if not selected_files:
                return {
                    "error": f"No files found from the specified list: {specific_files}"
                }
        elif specific_file:
            selected_files = self.select_specific_file(html_files, specific_file)
            if not selected_files:
                return {"error": f"Specific file not found: {specific_file}"}
        else:
            selected_files = self.select_random_files(html_files, file_count)

        # Validate each file pair
        validation_results = []
        successful_validations = 0
        total_coverage = 0

        print(f"\nValidating {len(selected_files)} file pairs...")
        print("-" * 60)

        for html_file in selected_files:
            rag_file = self.find_corresponding_rag_file(html_file)
            result = self.validate_file_pair(html_file, rag_file)
            validation_results.append(result)

            if result["validation_successful"]:
                successful_validations += 1
                total_coverage += result["comparison_results"]["coverage_percentage"]

        # Calculate summary statistics
        average_coverage = (
            total_coverage / successful_validations if successful_validations > 0 else 0
        )

        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_files_tested": len(selected_files),
            "successful_validations": successful_validations,
            "failed_validations": len(selected_files) - successful_validations,
            "average_coverage_percentage": round(average_coverage, 2),
            "validation_results": validation_results,
        }

        print("-" * 60)
        print(f"VALIDATION SUMMARY:")
        print(f"Total files tested: {summary['total_files_tested']}")
        print(f"Successful validations: {summary['successful_validations']}")
        print(f"Failed validations: {summary['failed_validations']}")
        print(f"Average coverage: {summary['average_coverage_percentage']}%")

        # Save results to JSON file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_report_file = self.report_path / f"validation_results_{timestamp}.json"

        with open(json_report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {json_report_file}")
        print("=" * 60)

        return summary

    def print_detailed_results(self, results: Dict[str, Any]):
        """Print detailed validation results."""
        print("\nDETAILED VALIDATION RESULTS:")
        print("=" * 80)

        for i, result in enumerate(results["validation_results"], 1):
            print(f"\n{i}. {result['html_file']}")
            print(f"   RAG File: {result['rag_file']}")

            if result["validation_successful"]:
                comp = result["comparison_results"]
                print(f"   ✓ Coverage: {comp['coverage_percentage']}%")
                print(f"   HTML Text Length: {comp['html_text_length']} chars")
                print(f"   RAG Text Length: {comp['rag_text_length']} chars")
                print(f"   Missing Sentences: {comp['missing_sentences_count']}")

                if comp["missing_sentences_count"] > 0:
                    print(f"   First few missing sentences:")
                    for j, sentence in enumerate(comp["missing_sentences"][:3]):
                        print(f"     - {sentence[:100]}...")
            else:
                print(f"   ✗ Error: {result['error']}")


def main():
    """Main function to run the validation test."""
    # Initialize and run validation
    validator = RAGValidationTest()
    results = validator.run_validation(file_count=10)

    # Print detailed results if validation was successful
    if "error" not in results:
        validator.print_detailed_results(results)
    else:
        print(f"Validation failed: {results['error']}")


if __name__ == "__main__":
    main()
