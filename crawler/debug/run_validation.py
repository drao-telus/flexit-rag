"""
Main script to run RAG validation with HTML report generation.

This script combines the validation test with HTML report generation
to provide a complete validation solution.
"""

import sys
from rag_processor import RAGProcessor
from html_report_generator import HTMLReportGenerator


def run_complete_validation(
    file_count: int = 10,
    generate_html: bool = True,
    specific_file: str = None,
    specific_files: list = None,
):
    """
    Run complete RAG validation with optional HTML report generation.

    Args:
        file_count (int): Number of files to validate (ignored if specific_file or specific_files is provided)
        generate_html (bool): Whether to generate HTML report
        specific_file (str): Specific file to validate (optional, for backward compatibility)
        specific_files (list): List of specific files to validate (optional)
    """
    print("Starting RAG Validation System")
    print("=" * 60)

    # Initialize validator
    validator = RAGProcessor()

    # Run validation
    if specific_files:
        print(f"Running validation for {len(specific_files)} specific files:")
        for file in specific_files:
            print(f"  - {file}")
        results = validator.run_validation(specific_files=specific_files)
    elif specific_file:
        print(f"Running validation for specific file: {specific_file}")
        results = validator.run_validation(specific_file=specific_file)
    else:
        print(f"Running validation for {file_count} random files")
        results = validator.run_validation(file_count=file_count)

    # Check if validation was successful
    if "error" in results:
        print(f"Validation failed: {results['error']}")
        return None

    # Print detailed results
    validator.print_detailed_results(results)

    # Generate HTML report if requested
    if generate_html:
        print("\n" + "=" * 60)
        print("GENERATING HTML REPORT")
        print("=" * 60)

        report_generator = HTMLReportGenerator()
        html_report_path = report_generator.generate_report(results)

        print(f"✓ HTML report generated: {html_report_path}")
        print("You can open this file in a web browser to view the detailed report.")

    return results


def main():
    """Main function with command line argument support."""
    file_count = 10
    generate_html = True
    specific_file = None
    specific_files = None

    # Parse command line arguments
    if len(sys.argv) > 1:
        # Check if first argument is a number (file count)
        try:
            file_count = int(sys.argv[1])
            # If it's a number, check for additional arguments
            if len(sys.argv) > 2:
                generate_html = sys.argv[2].lower() in ["true", "1", "yes", "y"]
        except ValueError:
            # If it's not a number, treat all arguments as filenames
            specific_files = sys.argv[1:]
            # Check if last argument might be generate_html flag
            if len(specific_files) > 1 and specific_files[-1].lower() in [
                "true",
                "1",
                "yes",
                "y",
                "false",
                "0",
                "no",
                "n",
            ]:
                generate_html = specific_files[-1].lower() in ["true", "1", "yes", "y"]
                specific_files = specific_files[:-1]  # Remove the flag from file list

            # If only one file provided, use backward compatibility
            if len(specific_files) == 1:
                specific_file = specific_files[0]
                specific_files = None
                print(f"Detected specific file: {specific_file}")
            else:
                print(f"Detected {len(specific_files)} specific files:")
                for i, file in enumerate(specific_files, 1):
                    print(f"  {i}. {file}")

    print(f"\nConfiguration:")
    if specific_files:
        print(f"- Multiple files to validate: {len(specific_files)}")
    elif specific_file:
        print(f"- Specific file to validate: {specific_file}")
    else:
        print(f"- Files to validate: {file_count} (random)")
    print(f"- Generate HTML report: {generate_html}")
    print()

    # Run validation
    results = run_complete_validation(
        file_count=file_count,
        generate_html=generate_html,
        specific_file=specific_file,
        specific_files=specific_files,
    )

    if results:
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(
            f"Summary: {results['successful_validations']}/{results['total_files_tested']} files validated successfully"
        )
        print(f"Average coverage: {results['average_coverage_percentage']}%")
    else:
        print("\n" + "=" * 60)
        print("VALIDATION FAILED")
        print("=" * 60)


if __name__ == "__main__":
    main()
