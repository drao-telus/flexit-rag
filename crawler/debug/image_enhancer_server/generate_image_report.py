import json
import os
from datetime import datetime


def generate_complete_image_report():
    """Generate a complete HTML image report from the JSON data"""

    # Read the JSON data
    json_path = os.path.join("crawler", "url", "image_mapping_cache.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data["metadata"]
    document_images = data["document_images"]

    # HTML template start
    html_content = f"""<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Mapping Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007bff;
        }}

        .header h1 {{
            color: #333;
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}

        .metadata {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}

        .metadata-item {{
            text-align: center;
        }}

        .metadata-item .value {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}

        .metadata-item .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .document-section {{
            margin-bottom: 50px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}

        .document-header {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            padding: 15px 20px;
            margin: 0;
        }}

        .document-header a {{
            color: white;
            text-decoration: none;
            font-size: 1.2em;
            font-weight: 500;
            word-break: break-all;
            display: block;
        }}

        .document-header a:hover {{
            text-decoration: underline;
        }}

        .images-container {{
            padding: 20px;
            background-color: #fafafa;
        }}

        .images-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }}

        .image-pair {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .image-pair:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }}

        .image-row {{
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }}

        .image-item {{
            flex: 1;
            text-align: center;
        }}

        .image-item img {{
            width: 150px;
            height: 150px;
            object-fit: cover;
            border: 2px solid #ddd;
            border-radius: 6px;
            transition: border-color 0.2s ease;
        }}

        .image-item img:hover {{
            border-color: #007bff;
        }}

        .image-label {{
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
            font-weight: 500;
        }}

        .image-description {{
            margin-top: 10px;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            font-size: 0.9em;
            color: #555;
            border-radius: 0 4px 4px 0;
        }}

        .no-images {{
            text-align: center;
            color: #888;
            font-style: italic;
            padding: 40px;
        }}

        .loading-placeholder {{
            width: 150px;
            height: 150px;
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
            border-radius: 6px;
        }}

        @keyframes loading {{
            0% {{
                background-position: 200% 0;
            }}

            100% {{
                background-position: -200% 0;
            }}
        }}

        .error-image {{
            width: 150px;
            height: 150px;
            background-color: #f8d7da;
            border: 2px solid #f5c6cb;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #721c24;
            font-size: 0.8em;
            text-align: center;
            padding: 5px;
            box-sizing: border-box;
        }}

        .progress-info {{
            background-color: #e3f2fd;
            border: 1px solid #2196f3;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 20px;
            text-align: center;
            color: #1976d2;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 15px;
            }}

            .images-grid {{
                grid-template-columns: 1fr;
            }}

            .image-row {{
                flex-direction: column;
                align-items: center;
            }}

            .metadata {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>

<body>
    <div class="container">
        <div class="header">
            <h1>Image Mapping Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>

        <div class="metadata">
            <div class="metadata-item">
                <span class="value">{metadata['total_documents']}</span>
                <span class="label">Total Documents</span>
            </div>
            <div class="metadata-item">
                <span class="value">{metadata['total_images']}</span>
                <span class="label">Total Images</span>
            </div>
            <div class="metadata-item">
                <span class="value">{metadata['documents_with_images']}</span>
                <span class="label">Documents with Images</span>
            </div>
            <div class="metadata-item">
                <span class="value">{metadata['errors_count']}</span>
                <span class="label">Errors</span>
            </div>
        </div>

        <div class="progress-info">
            <strong>Complete Report:</strong> Showing all {len(document_images)} documents with images
        </div>
"""

    # Generate sections for each document
    for doc_index, (document_url, images) in enumerate(document_images.items(), 1):
        html_content += f"""
        <div class="document-section">
            <h2 class="document-header">
                <a href="{document_url}" target="_blank">
                    {document_url}
                </a>
            </h2>
            <div class="images-container">
"""

        if images:
            html_content += '                <div class="images-grid">\n'

            for image in images:
                image_url = image.get("image_url", "")
                enhance_url = image.get("enhance_url", "")
                description = image.get("description", "No description available")

                html_content += f"""                    <div class="image-pair">
                        <div class="image-row">
                            <div class="image-item">
                                <img src="{image_url}" 
                                     alt="{description}"
                                     onerror="this.parentElement.innerHTML='<div class=\\'error-image\\'>Image not available<br><a href=\\'{image_url}\\' target=\\'_blank\\' style=\\'color: #721c24; text-decoration: underline;\\'>Debug URL</a></div>'">
                                <div class="image-label">Thumbnail</div>
                            </div>
                            <div class="image-item">
                                <img src="{enhance_url}" 
                                     alt="{description}"
                                     onerror="this.parentElement.innerHTML='<div class=\\'error-image\\'>Image not available<br><a href=\\'{enhance_url}\\' target=\\'_blank\\' style=\\'color: #721c24; text-decoration: underline;\\'>Debug URL</a></div>'">
                                <div class="image-label">Enhanced</div>
                            </div>
                        </div>
                        <div class="image-description">
                            <strong>Description:</strong> {description}<br>
                            <strong>Thumbnail URL:</strong> <a href="{image_url}" target="_blank" style="color: #007bff; word-break: break-all;">{image_url}</a><br>
                            <strong>Enhanced URL:</strong> <a href="{enhance_url}" target="_blank" style="color: #007bff; word-break: break-all;">{enhance_url}</a>
                        </div>
                    </div>
"""

            html_content += "                </div>\n"
        else:
            html_content += '                <div class="no-images">No images found for this document</div>\n'

        html_content += """            </div>
        </div>
"""

    # Close HTML
    html_content += """    </div>
</body>
</html>"""

    # Write the complete report
    output_path = os.path.join(
        "crawler", "debug", "image_enhancer_server" "image_report.html"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Complete image report generated successfully!")
    print(f"Report saved to: {output_path}")
    print(f"Total documents processed: {len(document_images)}")
    print(f"Total images: {metadata['total_images']}")


if __name__ == "__main__":
    generate_complete_image_report()
