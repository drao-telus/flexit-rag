# Image URL Enhancer Server

A Flask-based web application for debugging and editing image URLs in the RAG system. This tool provides an interactive interface to view, validate, and correct image URLs when they are not available or broken.

## Features

- **Interactive Web Interface**: View all images with editable URL fields
- **Debug Links**: Clickable URLs for troubleshooting broken images
- **Real-time Validation**: Check if image URLs are accessible
- **Bulk Operations**: Validate, preview, or reset all images at once
- **Data Persistence**: Save changes with automatic backup creation
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Server**:
   ```bash
   python crawler/debug/image_enhancer_server/app.py
   ```

3. **Access the Interface**:
   Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

### Viewing Images
- The main page displays all documents with their associated images
- Each image shows thumbnail and enhanced URL fields
- Broken images display debug links for troubleshooting

### Editing URLs
- Click on any URL field to edit it directly
- Use the "Save Changes" button to persist modifications
- Changes are automatically backed up before saving

### Validation
- Click "Validate URLs" to check if images are accessible
- Use "Validate All Images" for bulk validation
- Status indicators show validation results

### Preview
- "Preview" button opens images in new tabs for verification
- "Preview All Changes" shows all modified images

### Reset
- "Reset" button reverts unsaved changes
- "Reset All Changes" reverts all modifications

### Sorting
- "Sort by Missing Images" button moves documents with broken/missing images to the top
- "Reset Sort" button restores the original document order
- Sorting helps prioritize debugging efforts on problematic images

## API Endpoints

- `GET /` - Main interface
- `POST /api/update-image-url` - Update image URLs
- `POST /api/validate-image-url` - Validate image accessibility
- `GET /api/backups` - List available backups
- `POST /api/restore-backup` - Restore from backup

## File Structure

```
image_enhancer_server/
├── app.py                 # Flask application
├── requirements.txt       # Dependencies
├── generate_image_report.py # Original report generator
├── templates/
│   └── image_report.html  # Main interface template
└── static/
    └── js/
        └── image_editor.js # Client-side functionality
```

## Data Files

- `image_data.json` - Main image URL data
- `image_data_backup_*.json` - Automatic backups
- Generated from the RAG system's image mapping cache

## Keyboard Shortcuts

- `Ctrl+S` - Save changes
- `Esc` - Reset current changes

## Development

The server runs in debug mode by default. For production deployment, use a proper WSGI server like Gunicorn.

## Troubleshooting

1. **Server won't start**: Check if port 5000 is available
2. **Images not loading**: Verify the image URLs are accessible
3. **Changes not saving**: Check file permissions for JSON data files
4. **Validation failing**: Ensure internet connectivity for external image URLs
