from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
from datetime import datetime
import shutil
from pathlib import Path

app = Flask(__name__)

# Configuration
JSON_PATH = os.path.join("crawler", "url", "image_mapping_cache.json")
BACKUP_DIR = "backups"


def ensure_backup_dir():
    """Ensure backup directory exists"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)


def create_backup():
    """Create a backup of the current JSON file"""
    ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"image_mapping_cache_backup_{timestamp}.json"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)

    if os.path.exists(JSON_PATH):
        shutil.copy2(JSON_PATH, backup_path)
        return backup_path
    return None


def load_image_data():
    """Load image data from JSON file"""
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"metadata": {}, "document_images": {}}


def save_image_data(data):
    """Save image data to JSON file with backup"""
    create_backup()
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@app.route("/")
def index():
    """Serve the main image report page"""
    data = load_image_data()
    return render_template(
        "image_report.html",
        metadata=data.get("metadata", {}),
        document_images=data.get("document_images", {}),
        generation_time=datetime.now().strftime("%B %d, %Y at %I:%M %p"),
    )


@app.route("/api/update-image-url", methods=["POST"])
def update_image_url():
    """Update an image URL in the JSON cache"""
    try:
        request_data = request.get_json()
        document_url = request_data.get("document_url")
        image_index = request_data.get("image_index")
        url_type = request_data.get("url_type")  # 'image_url' or 'enhance_url'
        new_url = request_data.get("new_url")

        if not all([document_url, image_index is not None, url_type, new_url]):
            return (
                jsonify({"success": False, "error": "Missing required parameters"}),
                400,
            )

        # Load current data
        data = load_image_data()

        # Check if document exists
        if document_url not in data["document_images"]:
            return jsonify({"success": False, "error": "Document not found"}), 404

        # Check if image index is valid
        document_data = data["document_images"][document_url]
        images = document_data.get("images", [])
        if image_index >= len(images):
            return jsonify({"success": False, "error": "Image index out of range"}), 404

        # Update the URL
        if url_type in ["image_url", "enhance_url"]:
            images[image_index][url_type] = new_url

            # Save the updated data
            save_image_data(data)

            return jsonify(
                {
                    "success": True,
                    "message": f"{url_type} updated successfully",
                    "updated_url": new_url,
                }
            )
        else:
            return jsonify({"success": False, "error": "Invalid URL type"}), 400

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/validate-image-url", methods=["POST"])
def validate_image_url():
    """Validate if an image URL is accessible"""
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        request_data = request.get_json()
        url = request_data.get("url")

        if not url:
            return jsonify({"valid": False, "error": "No URL provided"})

        # Create a session with retry strategy
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Try to access the URL
        response = session.head(url, timeout=10, allow_redirects=True)

        if response.status_code == 200:
            content_type = response.headers.get("content-type", "").lower()
            if "image" in content_type:
                return jsonify({"valid": True, "content_type": content_type})
            else:
                return jsonify(
                    {
                        "valid": False,
                        "error": f"Not an image (content-type: {content_type})",
                    }
                )
        else:
            return jsonify({"valid": False, "error": f"HTTP {response.status_code}"})

    except requests.exceptions.RequestException as e:
        return jsonify({"valid": False, "error": str(e)})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


@app.route("/api/backups")
def list_backups():
    """List available backup files"""
    ensure_backup_dir()
    backups = []

    if os.path.exists(BACKUP_DIR):
        for filename in os.listdir(BACKUP_DIR):
            if filename.startswith("image_mapping_cache_backup_") and filename.endswith(
                ".json"
            ):
                filepath = os.path.join(BACKUP_DIR, filename)
                stat = os.stat(filepath)
                backups.append(
                    {
                        "filename": filename,
                        "created": datetime.fromtimestamp(stat.st_ctime).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "size": stat.st_size,
                    }
                )

    backups.sort(key=lambda x: x["created"], reverse=True)
    return jsonify({"backups": backups})


@app.route("/api/restore-backup", methods=["POST"])
def restore_backup():
    """Restore from a backup file"""
    try:
        request_data = request.get_json()
        backup_filename = request_data.get("backup_filename")

        if not backup_filename:
            return jsonify({"success": False, "error": "No backup filename provided"})

        backup_path = os.path.join(BACKUP_DIR, backup_filename)

        if not os.path.exists(backup_path):
            return jsonify({"success": False, "error": "Backup file not found"})

        # Create a backup of current state before restoring
        create_backup()

        # Restore the backup
        shutil.copy2(backup_path, JSON_PATH)

        return jsonify({"success": True, "message": f"Restored from {backup_filename}"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
