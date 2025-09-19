#!/usr/bin/env python3
"""
Semantic Enrichment Analyzer - Analyzes RAG output files to assess current semantic richness
and determine if semantic enrichment is needed.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set
import sys

sys.path.append(".")


class SemanticEnrichmentAnalyzer:
    """Analyzes RAG files to assess semantic richness and identify improvement opportunities."""

    def __init__(self, rag_output_dir: str = "crawler/result_data/rag_output"):
        self.rag_output_dir = rag_output_dir
        self.analysis_results = {}

    def analyze_all_files(self) -> Dict[str, Any]:
        """Analyze all RAG files and generate comprehensive semantic assessment."""

        print("🔍 Starting Semantic Enrichment Analysis...")
        print("=" * 60)

        # Get all RAG files
        rag_files = [
            f for f in os.listdir(self.rag_output_dir) if f.endswith("_rag.json")
        ]

        print(f"Found {len(rag_files)} RAG files to analyze")

        # Analyze each file
        file_analyses = []
        for rag_file in rag_files[:10]:  # Analyze first 10 files for sample
            file_path = os.path.join(self.rag_output_dir, rag_file)
            analysis = self._analyze_single_file(file_path)
            if analysis:
                file_analyses.append(analysis)

        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(file_analyses)

        # Generate recommendations
        recommendations = self._generate_recommendations(overall_assessment)

        return {
            "total_files_analyzed": len(file_analyses),
            "sample_files": [f["filename"] for f in file_analyses],
            "overall_assessment": overall_assessment,
            "recommendations": recommendations,
            "detailed_analysis": file_analyses,
        }

    def _analyze_single_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single RAG file for semantic richness."""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                rag_data = json.load(f)

            filename = Path(file_path).name

            # Extract current semantic information
            semantic_info = {
                "filename": filename,
                "document_id": rag_data.get("document_id", ""),
                "title": rag_data.get("title", ""),
                "breadcrumb": rag_data.get("breadcrumb", ""),
                "processing_strategy": rag_data.get("processing_strategy", ""),
                "total_chunks": rag_data.get("total_chunks", 0),
                "has_images": len(rag_data.get("images", [])) > 0,
                "image_count": len(rag_data.get("images", [])),
            }

            # Analyze chunks
            chunks = rag_data.get("chunks", [])
            chunk_analysis = self._analyze_chunks(chunks)

            # Analyze content structure
            content_analysis = self._analyze_content_structure(rag_data)

            # Assess semantic gaps
            semantic_gaps = self._identify_semantic_gaps(rag_data)

            return {
                **semantic_info,
                "chunk_analysis": chunk_analysis,
                "content_analysis": content_analysis,
                "semantic_gaps": semantic_gaps,
                "semantic_richness_score": self._calculate_semantic_score(
                    semantic_info, chunk_analysis, content_analysis
                ),
            }

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None

    def _analyze_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze chunk-level semantic information."""

        if not chunks:
            return {"error": "No chunks found"}

        chunk_types = [chunk.get("chunk_type", "") for chunk in chunks]
        chunk_strategies = [
            chunk.get("metadata", {}).get("chunking_strategy", "") for chunk in chunks
        ]

        # Analyze content features
        has_tables = any(
            chunk.get("metadata", {}).get("has_tables", False) for chunk in chunks
        )
        has_lists = any(
            chunk.get("metadata", {}).get("has_lists", False) for chunk in chunks
        )
        has_images = any(
            chunk.get("metadata", {}).get("has_images", False) for chunk in chunks
        )

        # Analyze content complexity
        word_counts = [
            chunk.get("metadata", {}).get("word_count", 0) for chunk in chunks
        ]
        avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0

        return {
            "chunk_count": len(chunks),
            "chunk_types": list(set(chunk_types)),
            "chunking_strategies": list(set(chunk_strategies)),
            "has_tables": has_tables,
            "has_lists": has_lists,
            "has_images": has_images,
            "avg_word_count": avg_word_count,
            "total_words": sum(word_counts),
        }

    def _analyze_content_structure(self, rag_data: Dict) -> Dict[str, Any]:
        """Analyze the structural and semantic elements of the content."""

        content = ""
        chunks = rag_data.get("chunks", [])
        for chunk in chunks:
            content += chunk.get("content", "") + "\n"

        # Analyze structural elements
        headings = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
        tables = content.count("|")  # Simple table detection
        lists = len(re.findall(r"^\d+\.\s+", content, re.MULTILINE))  # Numbered lists
        bullet_lists = len(re.findall(r"^-\s+", content, re.MULTILINE))  # Bullet lists

        # Analyze cross-references
        cross_refs = re.findall(r"\[([^\]]+)\]\([^)]+\)", content)  # Markdown links
        see_also_refs = re.findall(r"[Ss]ee\s+([^.]+)", content)

        # Analyze procedural content
        steps = len(re.findall(r"^\d+\.\s+", content, re.MULTILINE))
        instructions = len(
            re.findall(
                r"\b(click|select|choose|enter|configure)\b", content, re.IGNORECASE
            )
        )

        # Analyze domain terminology
        domain_terms = self._extract_domain_terms(content)

        return {
            "heading_count": len(headings),
            "headings": headings[:5],  # First 5 headings
            "table_indicators": tables > 10,  # Rough table detection
            "numbered_lists": lists,
            "bullet_lists": bullet_lists,
            "cross_references": len(cross_refs),
            "see_also_references": len(see_also_refs),
            "procedural_steps": steps,
            "instruction_count": instructions,
            "domain_terms": domain_terms,
            "content_length": len(content),
        }

    def _extract_domain_terms(self, content: str) -> List[str]:
        """Extract domain-specific terms that might benefit from glossary."""

        # Common domain terms in benefits/HR systems
        domain_patterns = [
            r"\b(beneficiary|beneficiaries)\b",
            r"\b(enrollment|enroll)\b",
            r"\b(coverage|benefit)\b",
            r"\b(premium|contribution)\b",
            r"\b(plan year|plan setup)\b",
            r"\b(employee|administrator)\b",
            r"\b(flex credit|excess flex)\b",
            r"\b(payroll|deduction)\b",
        ]

        found_terms = []
        for pattern in domain_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            found_terms.extend([m.lower() for m in matches])

        # Return unique terms with counts
        term_counts = Counter(found_terms)
        return [term for term, count in term_counts.most_common(10)]

    def _identify_semantic_gaps(self, rag_data: Dict) -> List[str]:
        """Identify potential semantic enrichment opportunities."""

        gaps = []

        # Check for missing breadcrumbs
        if not rag_data.get("breadcrumb"):
            gaps.append("Missing breadcrumb navigation")

        # Check for document classification
        title = rag_data.get("title", "").lower()
        if not any(
            doc_type in title
            for doc_type in ["guide", "reference", "setup", "configuration"]
        ):
            gaps.append("Document type not clearly identified")

        # Check for cross-references
        content = ""
        for chunk in rag_data.get("chunks", []):
            content += chunk.get("content", "")

        if "[" not in content or "](" not in content:
            gaps.append("No cross-references detected")

        # Check for procedural structure
        if "1." not in content and "step" not in content.lower():
            gaps.append("No clear procedural structure")

        # Check for topic tags
        gaps.append("No topic tags or categories")

        # Check for related documents
        gaps.append("No related document suggestions")

        return gaps

    def _calculate_semantic_score(
        self, semantic_info: Dict, chunk_analysis: Dict, content_analysis: Dict
    ) -> float:
        """Calculate a semantic richness score (0-100)."""

        score = 0

        # Basic metadata (20 points)
        if semantic_info.get("title"):
            score += 5
        if semantic_info.get("breadcrumb"):
            score += 5
        if semantic_info.get("processing_strategy"):
            score += 5
        if semantic_info.get("has_images"):
            score += 5

        # Content structure (30 points)
        if content_analysis.get("heading_count", 0) > 0:
            score += 10
        if content_analysis.get("cross_references", 0) > 0:
            score += 10
        if content_analysis.get("procedural_steps", 0) > 0:
            score += 10

        # Rich content features (30 points)
        if chunk_analysis.get("has_tables"):
            score += 10
        if chunk_analysis.get("has_lists"):
            score += 10
        if content_analysis.get("domain_terms"):
            score += 10

        # Advanced semantic features (20 points)
        # These would be added with semantic enrichment
        # Currently scoring 0 as they don't exist

        return min(score, 100)

    def _generate_overall_assessment(self, file_analyses: List[Dict]) -> Dict[str, Any]:
        """Generate overall assessment across all analyzed files."""

        if not file_analyses:
            return {"error": "No files to analyze"}

        # Calculate averages
        avg_semantic_score = sum(
            f.get("semantic_richness_score", 0) for f in file_analyses
        ) / len(file_analyses)

        # Count features
        files_with_breadcrumbs = sum(1 for f in file_analyses if f.get("breadcrumb"))
        files_with_cross_refs = sum(
            1
            for f in file_analyses
            if f.get("content_analysis", {}).get("cross_references", 0) > 0
        )
        files_with_procedures = sum(
            1
            for f in file_analyses
            if f.get("content_analysis", {}).get("procedural_steps", 0) > 0
        )

        # Common gaps
        all_gaps = []
        for f in file_analyses:
            all_gaps.extend(f.get("semantic_gaps", []))

        common_gaps = Counter(all_gaps).most_common(5)

        return {
            "average_semantic_score": avg_semantic_score,
            "files_with_breadcrumbs": files_with_breadcrumbs,
            "files_with_cross_references": files_with_cross_refs,
            "files_with_procedures": files_with_procedures,
            "most_common_gaps": common_gaps,
            "total_analyzed": len(file_analyses),
        }

    def _generate_recommendations(self, assessment: Dict) -> Dict[str, Any]:
        """Generate recommendations based on the assessment."""

        avg_score = assessment.get("average_semantic_score", 0)

        if avg_score >= 80:
            priority = "LOW"
            recommendation = "Your RAG files already have good semantic richness. Focus on other improvements."
        elif avg_score >= 60:
            priority = "MEDIUM"
            recommendation = (
                "Some semantic enrichment could be beneficial, but not critical."
            )
        else:
            priority = "HIGH"
            recommendation = "Significant semantic enrichment opportunities exist."

        # Specific recommendations based on gaps
        specific_recommendations = []
        common_gaps = assessment.get("most_common_gaps", [])

        for gap, count in common_gaps:
            if "breadcrumb" in gap.lower():
                specific_recommendations.append(
                    "Add breadcrumb navigation to improve context"
                )
            elif "cross-reference" in gap.lower():
                specific_recommendations.append(
                    "Implement cross-reference detection and linking"
                )
            elif "topic" in gap.lower():
                specific_recommendations.append("Add topic tagging and categorization")
            elif "document type" in gap.lower():
                specific_recommendations.append(
                    "Implement document type classification"
                )

        return {
            "priority": priority,
            "overall_recommendation": recommendation,
            "semantic_score": avg_score,
            "specific_recommendations": specific_recommendations,
            "estimated_effort": "Medium" if priority == "HIGH" else "Low",
        }

    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""

        analysis = self.analyze_all_files()

        report = []
        report.append("🔍 SEMANTIC ENRICHMENT ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")

        # Overall Assessment
        assessment = analysis["overall_assessment"]
        recommendations = analysis["recommendations"]

        report.append(f"📊 OVERALL ASSESSMENT")
        report.append(f"Files Analyzed: {analysis['total_files_analyzed']}")
        report.append(
            f"Average Semantic Score: {assessment['average_semantic_score']:.1f}/100"
        )
        report.append(
            f"Files with Breadcrumbs: {assessment['files_with_breadcrumbs']}/{assessment['total_analyzed']}"
        )
        report.append(
            f"Files with Cross-references: {assessment['files_with_cross_references']}/{assessment['total_analyzed']}"
        )
        report.append(
            f"Files with Procedures: {assessment['files_with_procedures']}/{assessment['total_analyzed']}"
        )
        report.append("")

        # Recommendations
        report.append(f"🎯 RECOMMENDATIONS")
        report.append(f"Priority: {recommendations['priority']}")
        report.append(f"Overall: {recommendations['overall_recommendation']}")
        report.append("")

        if recommendations["specific_recommendations"]:
            report.append("Specific Improvements:")
            for rec in recommendations["specific_recommendations"]:
                report.append(f"  • {rec}")
        report.append("")

        # Common Gaps
        report.append(f"🔍 MOST COMMON GAPS")
        for gap, count in assessment["most_common_gaps"]:
            report.append(f"  • {gap} ({count} files)")
        report.append("")

        # Sample file details
        report.append(f"📋 SAMPLE FILE ANALYSIS")
        for i, file_analysis in enumerate(analysis["detailed_analysis"][:3]):
            report.append(f"{i + 1}. {file_analysis['filename']}")
            report.append(
                f"   Score: {file_analysis['semantic_richness_score']:.1f}/100"
            )
            report.append(f"   Gaps: {', '.join(file_analysis['semantic_gaps'][:3])}")
            report.append("")

        return "\n".join(report)


def main():
    """Run the semantic enrichment analysis."""

    analyzer = SemanticEnrichmentAnalyzer()

    # Generate and print report
    report = analyzer.generate_report()
    print(report)

    # Save detailed analysis to file
    analysis = analyzer.analyze_all_files()

    output_file = "crawler/debug/semantic_analysis_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed analysis saved to: {output_file}")


if __name__ == "__main__":
    main()
