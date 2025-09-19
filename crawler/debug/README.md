# RAG Validation System

A simplified and focused RAG validation system that validates the completeness of RAG pipeline content extraction compared to original HTML files.

## Overview

This system provides a clean, basic implementation for validating RAG pipeline content completeness by:

1. **Random File Selection**: Selects ~10 HTML files randomly from `crawler/result_data/filtered_content`
2. **Text Extraction**: Uses BeautifulSoup with `soup.get_text(strip=True)` for clean text extraction from the `div id="mc-main-content"` section
3. **RAG Content Aggregation**: Handles multiple chunks in RAG JSON files by combining all chunk content
4. **Case-Insensitive Comparison**: Compares extracted HTML text with aggregated RAG content
5. **Missing Content Detection**: Identifies text present in HTML but missing from RAG output
6. **Comprehensive Reporting**: Generates both JSON and HTML reports with detailed analysis

## File Structure

```
crawler/debug/
├── text_extractor.py              # Text extraction utilities
├── rag_validation_test.py          # Main validation logic
├── html_report_generator.py        # HTML report creation
├── run_validation.py              # Main script to run validation
├── README.md                      # This documentation
└── report/                        # Output directory for reports
    ├── validation_results_[timestamp].json
    └── rag_validation_report_[timestamp].html
```

## Usage

### Basic Usage

Run validation with default settings (10 files):

```bash
python crawler/debug/run_validation.py
```

### Custom File Count

Validate a specific number of files:

```bash
python crawler/debug/run_validation.py 5
```

### Specific File Validation

Validate a specific HTML file by name:

```bash
# Validate a specific file (exact filename)
python crawler/debug/run_validation.py "specific_file.html"

# Validate a specific file with partial name matching
python crawler/debug/run_validation.py "partial_name"

# Validate specific file without HTML report
python crawler/debug/run_validation.py "specific_file.html" false
```

### Multiple File Validation

Validate multiple specific files in a single command:

```bash
# Multiple files
python crawler/debug/run_validation.py "file1.html" "file2.html" "file3.html"
```

**Important Notes for Windows Users:**
- Always use **double quotes** (`"`) around filenames, not single quotes (`'`)
- This is especially important for filenames containing spaces
- Single quotes will cause the command to split filenames incorrectly

### Without HTML Report

Generate only JSON report:

```bash
python crawler/debug/run_validation.py 10 false
```

### Advanced Usage Examples

```bash
# Validate 15 files with HTML report
python crawler/debug/run_validation.py 15 true

# Validate specific file with HTML report (default)
python crawler/debug/run_validation.py "example_page.html"

# Validate 5 files, then validate specific file, both with reports
python crawler/debug/run_validation.py 5
python crawler/debug/run_validation.py "problem_file.html"
```

### Individual Components

You can also run individual components:

```bash
# Run only validation test
python crawler/debug/rag_validation_test.py

# Generate HTML report from existing JSON
python crawler/debug/html_report_generator.py path/to/results.json
```

## Features

### 1. Text Extraction (`text_extractor.py`)

- **HTML Text Extraction**: Targets `div id="mc-main-content" role="main"` specifically
- **RAG Text Aggregation**: Combines content from multiple chunks in RAG JSON files
- **Markdown Cleaning**: Removes markdown formatting for accurate comparison
- **Case-Insensitive Comparison**: Ensures robust text matching
- **Missing Content Detection**: Identifies sentences present in HTML but missing from RAG

### 2. Validation Test (`rag_validation_test.py`)

- **Random File Selection**: Selects files randomly for unbiased testing
- **Specific File Selection**: Supports exact filename or partial name matching for targeted validation
- **File Pairing**: Automatically matches HTML files with corresponding RAG JSON files
- **Comprehensive Metrics**: Calculates coverage percentages and missing content statistics
- **Error Handling**: Gracefully handles missing files and processing errors
- **Progress Reporting**: Provides real-time feedback during validation

### 3. HTML Report Generator (`html_report_generator.py`)

- **Visual Reports**: Creates comprehensive HTML reports with interactive elements
- **Color-Coded Results**: Uses visual indicators for coverage levels
- **Missing Content Highlighting**: Shows exactly what content is missing
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Elements**: Expandable sections for detailed analysis

### 4. Main Runner (`run_validation.py`)

- **Command Line Interface**: Easy-to-use command line interface
- **Flexible Input Options**: Supports file count numbers or specific filenames
- **Configurable Options**: Supports custom file counts, specific files, and report generation options
- **Integrated Workflow**: Combines validation and reporting in one command
- **Smart Argument Parsing**: Automatically detects whether input is a file count or filename

## Output Reports

### JSON Report

Contains detailed validation results including:
- Timestamp and configuration
- Summary statistics (total files, success rate, average coverage)
- Individual file results with detailed comparison metrics
- Missing sentences for each file

### HTML Report

Provides a visual, interactive report with:
- Summary dashboard with key metrics
- File-by-file results with expandable details
- Color-coded coverage indicators
- Highlighted missing content sections
- Responsive design for easy viewing

## Technical Details

### Text Extraction Method

The system uses BeautifulSoup for robust HTML parsing:

```python
soup = BeautifulSoup(html_content, 'html.parser')
main_content_div = soup.find('div', {'id': 'mc-main-content', 'role': 'main'})
text = main_content_div.get_text(strip=True)
```

### RAG Content Handling

- Parses JSON files to extract all chunks
- Combines content from multiple chunks
- Handles both single and multi-chunk scenarios
- Removes markdown formatting for comparison

### Comparison Logic

- Case-insensitive text comparison
- Sentence-level analysis for detailed missing content detection
- Whitespace normalization
- 70% word match threshold for sentence coverage

## Requirements

- Python 3.7+
- BeautifulSoup4
- Standard library modules (json, datetime, pathlib, etc.)

## Installation

No additional installation required beyond the dependencies already in the project.

## Example Output

### Random File Validation

```
Configuration:
- Files to validate: 10
- Generate HTML report: True

Starting RAG Validation System
============================================================
Found 479 HTML files in filtered_content
Running validation for 10 random files
Randomly selected 10 files for validation

Validating 10 file pairs...
------------------------------------------------------------
✓ Validated File1.html - Coverage: 95.2%
✓ Validated File2.html - Coverage: 88.7%
...

VALIDATION SUMMARY:
Total files tested: 10
Successful validations: 10
Failed validations: 0
Average coverage: 91.5%

HTML report generated: crawler/debug/report/rag_validation_report_20250910_222945.html
```

### Specific File Validation

```
Configuration:
- Specific file to validate: example_page.html
- Generate HTML report: True

Starting RAG Validation System
============================================================
Running validation for specific file: example_page.html
Found matching file: example_page.html

Validating 1 file pair...
------------------------------------------------------------
✓ Validated example_page.html - Coverage: 92.3%

VALIDATION SUMMARY:
Total files tested: 1
Successful validations: 1
Failed validations: 0
Average coverage: 92.3%

HTML report generated: crawler/debug/report/rag_validation_report_20250910_223015.html
```

## Troubleshooting

### No HTML files found

Ensure you're running the script from the project root directory and that the `crawler/result_data/filtered_content` directory contains HTML files.

### Missing RAG files

The system will report missing RAG files in the validation results. Ensure corresponding `_rag.json` files exist in `crawler/result_data/rag_output`.

### Permission errors

Ensure the script has write permissions to create the `crawler/debug/report` directory and files within it.

## Comparison with Previous System

This simplified system replaces the complex `crawler/debug` implementation with:

- **Cleaner Architecture**: Focused, single-purpose modules
- **Better Documentation**: Clear code structure and comprehensive documentation
- **Improved Usability**: Simple command-line interface
- **Enhanced Reporting**: Visual HTML reports with interactive elements
- **Robust Text Extraction**: Specifically targets main content areas
- **Reliable Comparison**: Case-insensitive, whitespace-normalized comparison

The new system maintains all the core functionality while being much easier to understand, maintain, and extend.
