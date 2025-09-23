"""
Comprehensive HTML Report Generator

Creates a single HTML report with all files and highlighted missing sections.
Consolidates validation results into a comprehensive, visual report.
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, List
import html


class HTMLReportGenerator:
    """Generates comprehensive HTML reports for RAG validation results."""

    def __init__(self, report_path: str = "crawler/debug/report"):
        self.report_path = Path(report_path)
        self.report_path.mkdir(parents=True, exist_ok=True)

    def generate_report(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive HTML report from validation results.

        Args:
            validation_results (dict): Results from RAG validation test

        Returns:
            str: Path to the generated HTML report
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"rag_validation_report_{timestamp}.html"
        report_file_path = self.report_path / report_filename

        html_content = self._build_html_report(validation_results)

        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"HTML report generated: {report_file_path}")
        return str(report_file_path)

    def _build_html_report(self, results: Dict[str, Any]) -> str:
        """Build the complete HTML report content."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Validation Report - {results.get("timestamp", "Unknown")}</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="container">
        {self._build_header(results)}
        {self._build_summary_section(results)}
        {self._build_detailed_results(results)}
        {self._build_footer()}
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
"""
        return html_content

    def _get_css_styles(self) -> str:
        """Return CSS styles for the HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .summary-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .summary-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }
        
        .summary-card .number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .summary-card .label {
            color: #666;
            margin-top: 5px;
        }
        
        .results-section {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .results-header {
            background: #667eea;
            color: white;
            padding: 20px;
            font-size: 1.3em;
            font-weight: bold;
        }
        
        .file-result {
            border-bottom: 1px solid #eee;
            margin: 0;
        }
        
        .file-header {
            padding: 20px;
            cursor: pointer;
            background: #fafafa;
            border-left: 4px solid #ddd;
            transition: all 0.3s ease;
        }
        
        .file-header:hover {
            background: #f0f0f0;
        }
        
        .file-header.success {
            border-left-color: #28a745;
        }
        
        .file-header.warning {
            border-left-color: #ffc107;
        }
        
        .file-header.error {
            border-left-color: #dc3545;
        }
        
        .file-title {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 5px;
        }
        
        .file-stats {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        
        .stat {
            background: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            border: 1px solid #ddd;
        }
        
        .coverage-high { background: #d4edda; color: #155724; }
        .coverage-medium { background: #fff3cd; color: #856404; }
        .coverage-low { background: #f8d7da; color: #721c24; }
        
        .file-details {
            display: none;
            padding: 20px;
            background: #f9f9f9;
        }
        
        .file-details.show {
            display: block;
        }
        
        .missing-content {
            margin-top: 20px;
        }
        
        .missing-content h4 {
            color: #dc3545;
            margin-bottom: 10px;
        }
        
        .missing-sentence {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            font-family: monospace;
            font-size: 0.9em;
        }
        
        .toggle-icon {
            float: right;
            transition: transform 0.3s ease;
        }
        
        .toggle-icon.rotated {
            transform: rotate(180deg);
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        .content-comparison {
            margin-top: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .content-tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
        }
        
        .tab-button {
            flex: 1;
            padding: 10px 15px;
            border: none;
            background: transparent;
            cursor: pointer;
            font-size: 0.9em;
            transition: background-color 0.3s ease;
        }
        
        .tab-button:hover {
            background: #e9ecef;
        }
        
        .tab-button.active {
            background: #667eea;
            color: white;
        }
        
        .tab-content {
            display: none;
            padding: 15px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .content-text {
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            line-height: 1.4;
            white-space: pre-wrap;
            word-wrap: break-word;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 3px;
            border: 1px solid #e9ecef;
        }
        
        .json-content {
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 3px;
            border: 1px solid #e9ecef;
            overflow-x: auto;
        }
        
        .highlighted-missing {
            background: #ffeb3b;
            padding: 2px 4px;
            border-radius: 2px;
            font-weight: bold;
        }
        
        .markdown-content {
            background: #ffffff;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #e9ecef;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        
        .markdown-content h1,
        .markdown-content h2,
        .markdown-content h3,
        .markdown-content h4,
        .markdown-content h5,
        .markdown-content h6 {
            margin-top: 1.5em;
            margin-bottom: 0.5em;
            font-weight: 600;
            line-height: 1.25;
            color: #24292e;
        }
        
        .markdown-content h1 {
            font-size: 2em;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
        }
        
        .markdown-content h2 {
            font-size: 1.5em;
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
        }
        
        .markdown-content h3 {
            font-size: 1.25em;
        }
        
        .markdown-content h4 {
            font-size: 1em;
        }
        
        .markdown-content h5 {
            font-size: 0.875em;
        }
        
        .markdown-content h6 {
            font-size: 0.85em;
            color: #6a737d;
        }
        
        .markdown-content p {
            margin-bottom: 1em;
        }
        
        .markdown-content ul,
        .markdown-content ol {
            margin-bottom: 1em;
            padding-left: 2em;
        }
        
        .markdown-content li {
            margin-bottom: 0.25em;
        }
        
        .markdown-content blockquote {
            margin: 0 0 1em 0;
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e5;
        }
        
        .markdown-content code {
            background: #f6f8fa;
            border-radius: 3px;
            font-size: 85%;
            margin: 0;
            padding: 0.2em 0.4em;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }
        
        .markdown-content pre {
            background: #f6f8fa;
            border-radius: 6px;
            font-size: 85%;
            line-height: 1.45;
            overflow: auto;
            padding: 16px;
            margin-bottom: 1em;
        }
        
        .markdown-content pre code {
            background: transparent;
            border: 0;
            display: inline;
            line-height: inherit;
            margin: 0;
            max-width: auto;
            overflow: visible;
            padding: 0;
            word-wrap: normal;
        }
        
        .markdown-content table {
            border-collapse: collapse;
            border-spacing: 0;
            margin-bottom: 1em;
            width: 100%;
        }
        
        .markdown-content table th,
        .markdown-content table td {
            border: 1px solid #dfe2e5;
            padding: 6px 13px;
        }
        
        .markdown-content table th {
            background: #f6f8fa;
            font-weight: 600;
        }
        
        .markdown-content table tr:nth-child(2n) {
            background: #f6f8fa;
        }
        
        .markdown-content a {
            color: #0366d6;
            text-decoration: none;
        }
        
        .markdown-content a:hover {
            text-decoration: underline;
        }
        
        .markdown-content strong {
            font-weight: 600;
        }
        
        .markdown-content em {
            font-style: italic;
        }
        
        .markdown-content hr {
            background: #e1e4e8;
            border: 0;
            height: 0.25em;
            margin: 24px 0;
            padding: 0;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .file-stats {
                flex-direction: column;
                gap: 10px;
            }
            
            .content-tabs {
                flex-direction: column;
            }
            
            .tab-button {
                flex: none;
            }
        }
        """

    def _get_javascript(self) -> str:
        """Return JavaScript for interactive functionality."""
        return """
        function toggleFileDetails(element) {
            const details = element.nextElementSibling;
            const icon = element.querySelector('.toggle-icon');
            
            if (details.classList.contains('show')) {
                details.classList.remove('show');
                icon.classList.remove('rotated');
            } else {
                details.classList.add('show');
                icon.classList.add('rotated');
                
                // Render markdown content when details are shown
                renderMarkdownContent(details);
            }
        }
        
        function showTab(button, tabId) {
            // Get the parent container
            const container = button.closest('.content-comparison');
            
            // Hide all tab contents
            const tabContents = container.querySelectorAll('.tab-content');
            tabContents.forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all buttons
            const tabButtons = container.querySelectorAll('.tab-button');
            tabButtons.forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected tab and activate button
            const selectedTab = container.querySelector('#' + tabId);
            if (selectedTab) {
                selectedTab.classList.add('active');
                
                // If this is a RAG content tab, render markdown
                if (tabId.startsWith('rag-content')) {
                    renderMarkdownInTab(selectedTab);
                }
            }
            button.classList.add('active');
        }
        
        function renderMarkdownContent(detailsElement) {
            // Find all RAG content tabs in this details section
            const ragContentTabs = detailsElement.querySelectorAll('[id^="rag-content"]');
            ragContentTabs.forEach(tab => {
                renderMarkdownInTab(tab);
            });
        }
        
        function renderMarkdownInTab(tabElement) {
            // Check if marked.js is available
            if (typeof marked === 'undefined') {
                console.warn('marked.js library not loaded');
                return;
            }
            
            // Find the content text element
            const contentTextElement = tabElement.querySelector('.content-text');
            if (!contentTextElement) return;
            
            // Check if already rendered (avoid re-rendering)
            if (contentTextElement.classList.contains('markdown-rendered')) return;
            
            // Get the markdown text
            const markdownText = contentTextElement.textContent;
            
            // Configure marked options for better rendering
            marked.setOptions({
                breaks: true,
                gfm: true,
                headerIds: false,
                mangle: false
            });
            
            try {
                // Render markdown to HTML
                const htmlContent = marked.parse(markdownText);
                
                // Replace content with rendered HTML and apply markdown styling
                contentTextElement.innerHTML = htmlContent;
                contentTextElement.classList.remove('content-text');
                contentTextElement.classList.add('markdown-content', 'markdown-rendered');
            } catch (error) {
                console.error('Error rendering markdown:', error);
                // Keep original content if rendering fails
            }
        }
        
        // Add click handlers to all file headers
        document.addEventListener('DOMContentLoaded', function() {
            const headers = document.querySelectorAll('.file-header');
            headers.forEach(header => {
                header.addEventListener('click', function() {
                    toggleFileDetails(this);
                });
            });
            
            // Render markdown for any initially visible RAG content tabs
            const visibleRagTabs = document.querySelectorAll('[id^="rag-content"].active');
            visibleRagTabs.forEach(tab => {
                renderMarkdownInTab(tab);
            });
        });
        """

    def _build_header(self, results: Dict[str, Any]) -> str:
        """Build the header section of the report."""
        timestamp = results.get("timestamp", "Unknown")
        formatted_time = self._format_timestamp(timestamp)

        return f"""
        <div class="header">
            <h1>RAG Validation Report</h1>
            <div class="subtitle">Generated on {formatted_time}</div>
        </div>
        """

    def _build_summary_section(self, results: Dict[str, Any]) -> str:
        """Build the summary statistics section."""
        total_files = results.get("total_files_tested", 0)
        successful = results.get("successful_validations", 0)
        failed = results.get("failed_validations", 0)
        avg_coverage = results.get("average_coverage_percentage", 0)

        return f"""
        <div class="summary-section">
            <h2>Validation Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <div class="number">{total_files}</div>
                    <div class="label">Total Files Tested</div>
                </div>
                <div class="summary-card">
                    <div class="number">{successful}</div>
                    <div class="label">Successful Validations</div>
                </div>
                <div class="summary-card">
                    <div class="number">{failed}</div>
                    <div class="label">Failed Validations</div>
                </div>
                <div class="summary-card">
                    <div class="number">{avg_coverage}%</div>
                    <div class="label">Average Coverage</div>
                </div>
            </div>
        </div>
        """

    def _build_detailed_results(self, results: Dict[str, Any]) -> str:
        """Build the detailed results section."""
        validation_results = results.get("validation_results", [])

        if not validation_results:
            return """
            <div class="results-section">
                <div class="results-header">Detailed Results</div>
                <div style="padding: 20px;">
                    <div class="error-message">No validation results available.</div>
                </div>
            </div>
            """

        # Sort results by image issues first, then by coverage percentage (ascending order - lowest coverage first)
        def get_sort_key(result):
            if not result.get("validation_successful", False):
                return (
                    0,
                    -1,
                )  # Failed validations go first (0 = highest priority, -1 = lowest coverage)

            comparison = result.get("comparison_results", {})
            coverage = comparison.get("coverage_percentage", 0)

            # Get image analysis for sorting
            image_analysis = comparison.get("image_analysis", {})
            image_stats = image_analysis.get("image_statistics", {})

            total_content_refs = image_stats.get("total_content_references", 0)
            mapped_images = image_stats.get("successfully_mapped", 0)

            # Calculate image issue priority
            if total_content_refs == 0:
                image_priority = (
                    3  # No images needed - lowest priority for image issues
                )
            elif mapped_images < total_content_refs:
                image_priority = 1  # Has missing images - highest priority
            else:
                image_priority = 2  # All images mapped - medium priority

            return (
                image_priority,
                coverage,
            )  # Sort by image issues first, then coverage

        sorted_results = sorted(validation_results, key=get_sort_key)

        results_html = """
        <div class="results-section">
            <div class="results-header">Detailed Results (Ordered by Image Issues, then Coverage: Lowest to Highest)</div>
        """

        for i, result in enumerate(sorted_results, 1):
            results_html += self._build_file_result(i, result)

        results_html += "</div>"
        return results_html

    def _build_file_result(self, index: int, result: Dict[str, Any]) -> str:
        """Build HTML for a single file validation result."""
        html_file = result.get("html_file", "Unknown")
        rag_file = result.get("rag_file", "Unknown")
        successful = result.get("validation_successful", False)
        error = result.get("error")
        comparison = result.get("comparison_results", {})

        # Determine status class
        if not successful:
            status_class = "error"
            coverage_text = "Failed"
        else:
            coverage = comparison.get("coverage_percentage", 0)
            if coverage >= 90:
                status_class = "success"
            elif coverage >= 70:
                status_class = "warning"
            else:
                status_class = "error"
            coverage_text = f"{coverage}%"

        # Get image analysis for header
        image_analysis = comparison.get("image_analysis", {}) if successful else {}
        image_stats = image_analysis.get("image_statistics", {})

        # Calculate image coverage percentage
        total_content_refs = image_stats.get("total_content_references", 0)
        mapped_images = image_stats.get("successfully_mapped", 0)
        image_coverage_percent = (
            (mapped_images / total_content_refs * 100)
            if total_content_refs > 0
            else 100
        )

        # Determine image coverage class
        if total_content_refs == 0:
            image_coverage_class = "high"  # No images needed
            image_coverage_text = "N/A"
        elif image_coverage_percent == 100:
            image_coverage_class = "high"
            image_coverage_text = f"{image_coverage_percent:.0f}%"
        elif image_coverage_percent >= 80:
            image_coverage_class = "medium"
            image_coverage_text = f"{image_coverage_percent:.0f}%"
        else:
            image_coverage_class = "low"
            image_coverage_text = f"{image_coverage_percent:.0f}%"

        # Build file header
        file_html = f"""
        <div class="file-result">
            <div class="file-header {status_class}">
                <div class="file-title">{index}. {html.escape(html_file)}</div>
                <div class="file-stats">
                    <span class="stat">RAG File: {html.escape(rag_file)}</span>
                    <span class="stat coverage-{self._get_coverage_class(comparison.get("coverage_percentage", 0))}">
                        Coverage: {coverage_text}
                    </span>
                    <span class="stat coverage-{image_coverage_class}">
                        Images: {image_coverage_text} ({mapped_images}/{total_content_refs})
                    </span>
                </div>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="file-details">
        """

        # Build file details
        if successful:
            file_html += self._build_successful_result_details(comparison, index)
        else:
            file_html += f"""
                <div class="error-message">
                    <strong>Validation Error:</strong> {html.escape(str(error))}
                </div>
            """

        file_html += """
            </div>
        </div>
        """

        return file_html

    def _build_successful_result_details(
        self, comparison: Dict[str, Any], index: int
    ) -> str:
        """Build details for a successful validation result."""
        html_length = comparison.get("html_text_length", 0)
        rag_length = comparison.get("rag_text_length", 0)
        html_sentences = comparison.get("html_sentences_count", 0)
        missing_count = comparison.get("missing_sentences_count", 0)
        missing_sentences = comparison.get("missing_sentences", [])
        coverage = comparison.get("coverage_percentage", 0)
        html_text = comparison.get("html_text", "")
        rag_text = comparison.get("rag_text", "")
        rag_json = comparison.get("rag_json", {})

        details_html = f"""
        <div class="file-stats">
            <span class="stat">HTML Text: {html_length:,} characters</span>
            <span class="stat">RAG Text: {rag_length:,} characters</span>
            <span class="stat">HTML Sentences: {html_sentences}</span>
            <span class="stat">Missing Sentences: {missing_count}</span>
        </div>
        """

        # Generate unique IDs for this file's tabs
        unique_suffix = f"-{index}"
        rag_content_id = f"rag-content{unique_suffix}"
        rag_json_id = f"rag-json{unique_suffix}"
        html_content_id = f"html-content{unique_suffix}"
        topics_content_id = f"topics-content{unique_suffix}"
        image_analysis_id = f"image-analysis{unique_suffix}"

        # Add content comparison section
        details_html += f"""
        <div class="content-comparison">
            <div class="content-tabs">
                <button class="tab-button active" onclick="showTab(this, '{rag_content_id}')">RAG Content</button>
                <button class="tab-button" onclick="showTab(this, '{rag_json_id}')">RAG JSON</button>
                <button class="tab-button" onclick="showTab(this, '{html_content_id}')">HTML Content</button>
                <button class="tab-button" onclick="showTab(this, '{topics_content_id}')">Topics</button>
                <button class="tab-button" onclick="showTab(this, '{image_analysis_id}')">Image Analysis</button>
            </div>
        """

        # RAG content tab (now first and active)
        rag_markdown_content = self._get_rag_markdown_content(rag_json)
        details_html += f"""
            <div id="{rag_content_id}" class="tab-content active">
                <h4>RAG Content (Markdown Format)</h4>
                <div class="content-text">{html.escape(rag_markdown_content)}</div>
            </div>
        """

        # RAG JSON tab (now second)
        rag_json_str = json.dumps(rag_json, indent=2, ensure_ascii=False)
        details_html += f"""
            <div id="{rag_json_id}" class="tab-content">
                <h4>RAG JSON Structure</h4>
                <pre class="json-content">{html.escape(rag_json_str)}</pre>
            </div>
        """

        # HTML content tab (now third)
        highlighted_html = self._highlight_missing_content(html_text, missing_sentences)
        details_html += f"""
            <div id="{html_content_id}" class="tab-content">
                <h4>Extracted HTML Text</h4>
                <div class="content-text">{highlighted_html}</div>
            </div>
        """

        # Topics tab (fourth tab)
        topics_content = self._get_topics_content(rag_json)
        details_html += f"""
            <div id="{topics_content_id}" class="tab-content">
                <h4>Extracted Topics</h4>
                <div class="content-text">{topics_content}</div>
            </div>
        """

        # Image Analysis tab (fifth tab)
        image_analysis_content = self._get_image_analysis_content(comparison)
        details_html += f"""
            <div id="{image_analysis_id}" class="tab-content">
                <h4>Image Analysis</h4>
                <div class="content-text">{image_analysis_content}</div>
            </div>
        </div>
        """

        if missing_sentences:
            details_html += """
            <div class="missing-content">
                <h4>Missing Content (First 5 sentences):</h4>
            """

            for sentence in missing_sentences[:5]:
                escaped_sentence = html.escape(sentence[:200])
                if len(sentence) > 200:
                    escaped_sentence += "..."
                details_html += (
                    f'<div class="missing-sentence">{escaped_sentence}</div>'
                )

            if len(missing_sentences) > 5:
                details_html += f"<p><em>... and {len(missing_sentences) - 5} more missing sentences</em></p>"

            details_html += "</div>"
        else:
            details_html += (
                "<p><em>No missing content detected - excellent coverage!</em></p>"
            )

        return details_html

    def _get_coverage_class(self, coverage: float) -> str:
        """Get CSS class based on coverage percentage."""
        if coverage >= 90:
            return "high"
        elif coverage >= 70:
            return "medium"
        else:
            return "low"

    def _format_timestamp(self, timestamp: str) -> str:
        """Format ISO timestamp to readable format."""
        try:
            dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%B %d, %Y at %I:%M %p")
        except:
            return timestamp

    def _highlight_missing_content(
        self, html_text: str, missing_sentences: List[str]
    ) -> str:
        """Highlight missing content in the HTML text."""
        if not missing_sentences:
            return html.escape(html_text)

        highlighted_text = html.escape(html_text)

        # Sort missing sentences by length (longest first) to avoid partial replacements
        sorted_missing = sorted(missing_sentences, key=len, reverse=True)

        for sentence in sorted_missing:
            escaped_sentence = html.escape(sentence)
            highlighted_sentence = (
                f'<span class="highlighted-missing">{escaped_sentence}</span>'
            )
            highlighted_text = highlighted_text.replace(
                escaped_sentence, highlighted_sentence
            )

        return highlighted_text

    def _get_rag_markdown_content(self, rag_json: Dict[str, Any]) -> str:
        """Extract markdown content from RAG JSON."""
        try:
            chunks = rag_json.get("chunks", [])
            if not chunks:
                return "No content available"

            # Combine all chunk content (which is already in markdown format)
            markdown_content = ""
            for chunk in chunks:
                content = chunk.get("content", "")
                if content:
                    markdown_content += content + "\n\n"

            return markdown_content.strip()
        except Exception as e:
            return f"Error extracting markdown content: {str(e)}"

    def _get_topics_content(self, rag_json: Dict[str, Any]) -> str:
        """Extract and format topics from RAG JSON."""
        try:
            # Try to get topics from metadata first
            metadata = rag_json.get("metadata", {})
            topics = metadata.get("topics", {})

            # If not in metadata, try chunks
            if not topics:
                chunks = rag_json.get("chunks", [])
                if chunks:
                    chunk_metadata = chunks[0].get("metadata", {})
                    topics = chunk_metadata.get("topics", {})

            if not topics:
                return "No topics available"

            # Format topics nicely
            topics_text = ""

            # Breadcrumb topics
            breadcrumb_topics = topics.get("breadcrumb_topics", [])
            if breadcrumb_topics:
                topics_text += "BREADCRUMB TOPICS:\n"
                topics_text += ", ".join(breadcrumb_topics) + "\n\n"

            # Primary topics
            primary_topics = topics.get("primary_topics", [])
            if primary_topics:
                topics_text += "PRIMARY TOPICS:\n"
                topics_text += ", ".join(primary_topics) + "\n\n"

            # Content topics
            content_topics = topics.get("content_topics", [])
            if content_topics:
                topics_text += "CONTENT TOPICS:\n"
                topics_text += ", ".join(content_topics) + "\n\n"

            # Heading topics
            heading_topics = topics.get("heading_topics", [])
            if heading_topics:
                topics_text += "HEADING TOPICS:\n"
                topics_text += ", ".join(heading_topics) + "\n\n"

            # Keyword topics
            keyword_topics = topics.get("keyword_topics", [])
            if keyword_topics:
                topics_text += "KEYWORD TOPICS:\n"
                topics_text += ", ".join(keyword_topics) + "\n\n"

            # Entity topics
            entity_topics = topics.get("entity_topics", [])
            if entity_topics:
                topics_text += "ENTITY TOPICS:\n"
                topics_text += ", ".join(entity_topics) + "\n\n"

            # All topics
            all_topics = topics.get("all_topics", [])
            if all_topics:
                topics_text += "ALL TOPICS:\n"
                topics_text += ", ".join(all_topics) + "\n\n"

            # Topic count
            topic_count = topics.get("topic_count", 0)
            if topic_count:
                topics_text += f"TOTAL TOPIC COUNT: {topic_count}\n"

            return topics_text.strip() if topics_text else "No topics found"

        except Exception as e:
            return f"Error extracting topics: {str(e)}"

    def _get_image_analysis_content(self, comparison: Dict[str, Any]) -> str:
        """Extract and format image analysis information."""
        try:
            image_analysis = comparison.get("image_analysis", {})
            if not image_analysis:
                return "No image analysis available"

            image_stats = image_analysis.get("image_statistics", {})

            # Build image analysis content
            content = ""

            # Basic statistics
            content += "=== IMAGE ANALYSIS SUMMARY ===\n"
            content += f"Document Category: {image_analysis.get('document_category', 'Unknown')}\n"
            content += (
                f"Images in RAG array: {image_stats.get('total_images_in_array', 0)}\n"
            )
            content += f"Content image references: {image_stats.get('total_content_references', 0)}\n"
            content += (
                f"Successfully mapped: {image_stats.get('successfully_mapped', 0)}\n"
            )
            content += f"Missing images: {image_stats.get('missing_count', 0)}\n"
            content += f"Mapping success rate: {image_stats.get('mapping_success_rate', 0)}%\n\n"

            # Images in array
            images_in_array = image_analysis.get("images_in_array", [])
            if images_in_array:
                content += "=== IMAGES IN RAG ARRAY ===\n"
                for i, img in enumerate(images_in_array, 1):
                    content += f"{i}. {img.get('filename', 'Unknown')}\n"
                    content += f"   Description: {img.get('description', 'N/A')}\n"
                    content += f"   URL: {img.get('image_url', 'N/A')}\n"
                    content += f"   Local Path: {img.get('local_path', 'N/A')}\n\n"
            else:
                content += (
                    "=== IMAGES IN RAG ARRAY ===\nNo images found in RAG array\n\n"
                )

            # Content references
            content_refs = image_analysis.get("content_image_references", [])
            if content_refs:
                content += "=== CONTENT IMAGE REFERENCES ===\n"
                for i, ref in enumerate(content_refs, 1):
                    content += f"{i}. {ref}\n"
                content += "\n"
            else:
                content += "=== CONTENT IMAGE REFERENCES ===\nNo image references found in content\n\n"

            # Successfully mapped images
            mapped_images = image_analysis.get("mapped_images", [])
            if mapped_images:
                content += "=== SUCCESSFULLY MAPPED IMAGES ===\n"
                for i, img in enumerate(mapped_images, 1):
                    content += f"{i}. Reference: {img.get('reference', 'Unknown')}\n"
                    content += f"   Filename: {img.get('filename', 'Unknown')}\n"
                    content += f"   Path: {img.get('path', 'Unknown')}\n"
                    content += f"   Status: {img.get('status', 'Unknown')}\n\n"
            else:
                content += "=== SUCCESSFULLY MAPPED IMAGES ===\nNo successfully mapped images\n\n"

            # Missing images
            missing_images = image_analysis.get("missing_images", [])
            if missing_images:
                content += "=== MISSING IMAGES ===\n"
                for i, img in enumerate(missing_images, 1):
                    content += f"{i}. Reference: {img.get('reference', 'Unknown')}\n"
                    content += f"   Filename: {img.get('filename', 'Unknown')}\n"
                    content += f"   Status: {img.get('status', 'Unknown')}\n"
                    content += f"   Reason: {img.get('reason', 'Unknown')}\n"
                    if img.get("path"):
                        content += f"   Path: {img.get('path')}\n"
                    content += "\n"
            else:
                content += "=== MISSING IMAGES ===\nNo missing images - all content references are mapped!\n\n"

            return content.strip()

        except Exception as e:
            return f"Error extracting image analysis: {str(e)}"

    def _build_footer(self) -> str:
        """Build the footer section."""
        return """
        <div class="footer">
            <p>Generated by RAG Validation System</p>
            <p>This report shows the completeness of RAG pipeline content extraction compared to original HTML files.</p>
        </div>
        """


def generate_html_report_from_json(json_file_path: str) -> str:
    """
    Generate HTML report from a JSON validation results file.

    Args:
        json_file_path (str): Path to the JSON results file

    Returns:
        str: Path to the generated HTML report
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        generator = HTMLReportGenerator()
        return generator.generate_report(results)

    except Exception as e:
        print(f"Error generating HTML report from JSON: {e}")
        return ""


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        html_report = generate_html_report_from_json(json_file)
        if html_report:
            print(f"HTML report generated: {html_report}")
    else:
        print("Usage: python html_report_generator.py <json_results_file>")
