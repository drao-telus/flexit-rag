// Image Editor JavaScript for Interactive Image URL Editing

// Global variables
let isEditing = false;
let currentEditingPair = null;

// Utility functions
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    // Show notification
    setTimeout(() => notification.classList.add('show'), 100);

    // Hide notification after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => document.body.removeChild(notification), 300);
    }, 3000);
}

function addStatusIndicator(element, status) {
    // Remove existing status indicators
    const existingIndicators = element.querySelectorAll('.status-indicator');
    existingIndicators.forEach(indicator => indicator.remove());

    // Add new status indicator
    const indicator = document.createElement('span');
    indicator.className = `status-indicator status-${status}`;
    element.appendChild(indicator);
}

function getImagePairData(button) {
    const imagePair = button.closest('.image-pair');
    const documentUrl = imagePair.dataset.documentUrl;
    const imageIndex = parseInt(imagePair.dataset.imageIndex);
    const thumbnailInput = imagePair.querySelector('.thumbnail-url');
    const enhancedInput = imagePair.querySelector('.enhanced-url');

    return {
        imagePair,
        documentUrl,
        imageIndex,
        thumbnailInput,
        enhancedInput,
        thumbnailUrl: thumbnailInput.value.trim(),
        enhancedUrl: enhancedInput.value.trim()
    };
}

// Main functions
async function saveImageUrls(button) {
    const data = getImagePairData(button);
    const saveBtn = button;

    // Disable button during save
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';
    addStatusIndicator(saveBtn, 'loading');

    try {
        // Save thumbnail URL
        if (data.thumbnailUrl !== data.thumbnailInput.dataset.originalValue) {
            const thumbnailResponse = await fetch('/api/update-image-url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    document_url: data.documentUrl,
                    image_index: data.imageIndex,
                    url_type: 'image_url',
                    new_url: data.thumbnailUrl
                })
            });

            const thumbnailResult = await thumbnailResponse.json();
            if (!thumbnailResult.success) {
                throw new Error(`Thumbnail URL: ${thumbnailResult.error}`);
            }
        }

        // Save enhanced URL
        if (data.enhancedUrl !== data.enhancedInput.dataset.originalValue) {
            const enhancedResponse = await fetch('/api/update-image-url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    document_url: data.documentUrl,
                    image_index: data.imageIndex,
                    url_type: 'enhance_url',
                    new_url: data.enhancedUrl
                })
            });

            const enhancedResult = await enhancedResponse.json();
            if (!enhancedResult.success) {
                throw new Error(`Enhanced URL: ${enhancedResult.error}`);
            }
        }

        // Update original values
        data.thumbnailInput.dataset.originalValue = data.thumbnailUrl;
        data.enhancedInput.dataset.originalValue = data.enhancedUrl;

        // Show success
        addStatusIndicator(saveBtn, 'success');
        showNotification('Image URLs saved successfully!', 'success');

    } catch (error) {
        console.error('Error saving URLs:', error);
        addStatusIndicator(saveBtn, 'error');
        showNotification(`Error saving URLs: ${error.message}`, 'error');
    } finally {
        // Re-enable button
        saveBtn.disabled = false;
        saveBtn.textContent = 'Save Changes';
    }
}

function resetImageUrls(button) {
    const data = getImagePairData(button);

    // Reset to original values
    data.thumbnailInput.value = data.thumbnailInput.dataset.originalValue;
    data.enhancedInput.value = data.enhancedInput.dataset.originalValue;

    showNotification('URLs reset to original values', 'success');
}

async function validateImageUrls(button) {
    const data = getImagePairData(button);
    const validateBtn = button;

    // Disable button during validation
    validateBtn.disabled = true;
    validateBtn.textContent = 'Validating...';
    addStatusIndicator(validateBtn, 'loading');

    try {
        const results = [];

        // Validate thumbnail URL
        if (data.thumbnailUrl) {
            const thumbnailResponse = await fetch('/api/validate-image-url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: data.thumbnailUrl })
            });

            const thumbnailResult = await thumbnailResponse.json();
            results.push({
                type: 'Thumbnail',
                url: data.thumbnailUrl,
                valid: thumbnailResult.valid,
                error: thumbnailResult.error,
                contentType: thumbnailResult.content_type
            });
        }

        // Validate enhanced URL
        if (data.enhancedUrl) {
            const enhancedResponse = await fetch('/api/validate-image-url', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: data.enhancedUrl })
            });

            const enhancedResult = await enhancedResponse.json();
            results.push({
                type: 'Enhanced',
                url: data.enhancedUrl,
                valid: enhancedResult.valid,
                error: enhancedResult.error,
                contentType: enhancedResult.content_type
            });
        }

        // Show validation results
        const validResults = results.filter(r => r.valid);
        const invalidResults = results.filter(r => !r.valid);

        if (invalidResults.length === 0) {
            addStatusIndicator(validateBtn, 'success');
            showNotification(`All URLs are valid! (${validResults.length} checked)`, 'success');
        } else {
            addStatusIndicator(validateBtn, 'error');
            const errorMessages = invalidResults.map(r => `${r.type}: ${r.error}`).join(', ');
            showNotification(`Validation failed: ${errorMessages}`, 'error');
        }

    } catch (error) {
        console.error('Error validating URLs:', error);
        addStatusIndicator(validateBtn, 'error');
        showNotification(`Error validating URLs: ${error.message}`, 'error');
    } finally {
        // Re-enable button
        validateBtn.disabled = false;
        validateBtn.textContent = 'Validate URLs';
    }
}

function previewImages(button) {
    const data = getImagePairData(button);

    // Get the image elements
    const thumbnailImg = data.imagePair.querySelector('.image-item:first-child img');
    const enhancedImg = data.imagePair.querySelector('.image-item:last-child img');

    // Update image sources for preview
    if (data.thumbnailUrl && data.thumbnailUrl !== thumbnailImg.src) {
        thumbnailImg.src = data.thumbnailUrl;
        thumbnailImg.onerror = function () {
            this.parentElement.innerHTML = `<div class='error-image'>Image not available<br><a href='${data.thumbnailUrl}' target='_blank' style='color: #721c24; text-decoration: underline;'>Debug URL</a></div>`;
        };
    }

    if (data.enhancedUrl && data.enhancedUrl !== enhancedImg.src) {
        enhancedImg.src = data.enhancedUrl;
        enhancedImg.onerror = function () {
            this.parentElement.innerHTML = `<div class='error-image'>Image not available<br><a href='${data.enhancedUrl}' target='_blank' style='color: #721c24; text-decoration: underline;'>Debug URL</a></div>`;
        };
    }

    showNotification('Images updated with new URLs for preview', 'success');
}

// Auto-save functionality (optional)
function setupAutoSave() {
    const urlInputs = document.querySelectorAll('.url-input');

    urlInputs.forEach(input => {
        let saveTimeout;

        input.addEventListener('input', function () {
            // Clear existing timeout
            if (saveTimeout) {
                clearTimeout(saveTimeout);
            }

            // Set new timeout for auto-save (3 seconds after user stops typing)
            saveTimeout = setTimeout(() => {
                const imagePair = this.closest('.image-pair');
                const saveBtn = imagePair.querySelector('.save-btn');

                // Only auto-save if the value has changed
                if (this.value.trim() !== this.dataset.originalValue) {
                    console.log('Auto-saving changes...');
                    // Uncomment the line below to enable auto-save
                    // saveImageUrls(saveBtn);
                }
            }, 3000);
        });
    });
}

// Keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function (e) {
        // Ctrl+S to save current editing pair
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();

            const focusedInput = document.activeElement;
            if (focusedInput && focusedInput.classList.contains('url-input')) {
                const imagePair = focusedInput.closest('.image-pair');
                const saveBtn = imagePair.querySelector('.save-btn');
                saveImageUrls(saveBtn);
            }
        }

        // Escape to reset current editing pair
        if (e.key === 'Escape') {
            const focusedInput = document.activeElement;
            if (focusedInput && focusedInput.classList.contains('url-input')) {
                const imagePair = focusedInput.closest('.image-pair');
                const resetBtn = imagePair.querySelector('.reset-btn');
                resetImageUrls(resetBtn);
            }
        }
    });
}

// Bulk operations
function setupBulkOperations() {
    // Add bulk operation buttons to the header
    const progressInfo = document.querySelector('.progress-info');

    const bulkControls = document.createElement('div');
    bulkControls.style.marginTop = '10px';
    bulkControls.innerHTML = `
        <button class="btn btn-primary" onclick="validateAllImages()">Validate All Images</button>
        <button class="btn btn-success" onclick="previewAllImages()">Preview All Changes</button>
        <button class="btn btn-warning" onclick="resetAllImages()">Reset All Changes</button>
    `;

    progressInfo.appendChild(bulkControls);
}

async function validateAllImages() {
    const validateBtns = document.querySelectorAll('.validate-btn');
    showNotification('Starting bulk validation...', 'success');

    for (let i = 0; i < validateBtns.length; i++) {
        await validateImageUrls(validateBtns[i]);
        // Small delay between validations to avoid overwhelming the server
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    showNotification('Bulk validation completed!', 'success');
}

function previewAllImages() {
    const previewBtns = document.querySelectorAll('.preview-btn');
    previewBtns.forEach(btn => previewImages(btn));
    showNotification('All images updated for preview!', 'success');
}

function resetAllImages() {
    const resetBtns = document.querySelectorAll('.reset-btn');
    resetBtns.forEach(btn => resetImageUrls(btn));
    showNotification('All URLs reset to original values!', 'success');
}

// Sorting functionality
let originalOrder = [];
let isSorted = false;

function hasMissingImages(documentSection) {
    // Check if any images in this document have missing/broken URLs
    const imagePairs = documentSection.querySelectorAll('.image-pair');

    for (let pair of imagePairs) {
        const thumbnailInput = pair.querySelector('.thumbnail-url');
        const enhancedInput = pair.querySelector('.enhanced-url');

        const thumbnailUrl = thumbnailInput ? thumbnailInput.value.trim() : '';
        const enhancedUrl = enhancedInput ? enhancedInput.value.trim() : '';

        // Check for empty URLs or common error indicators
        if (!thumbnailUrl || !enhancedUrl ||
            thumbnailUrl === '' || enhancedUrl === '' ||
            thumbnailUrl.includes('404') || enhancedUrl.includes('404') ||
            thumbnailUrl.toLowerCase().includes('error') || enhancedUrl.toLowerCase().includes('error') ||
            thumbnailUrl.toLowerCase().includes('not-found') || enhancedUrl.toLowerCase().includes('not-found') ||
            thumbnailUrl.toLowerCase().includes('missing') || enhancedUrl.toLowerCase().includes('missing')) {
            return true;
        }

        // Check if images are actually broken by looking for error-image divs
        const errorImages = pair.querySelectorAll('.error-image');
        if (errorImages.length > 0) {
            return true;
        }
    }

    return false;
}

function sortByMissingImages() {
    const container = document.querySelector('.container');
    const documentSections = Array.from(container.querySelectorAll('.document-section'));

    // Store original order if not already stored
    if (originalOrder.length === 0) {
        originalOrder = documentSections.map(section => section.cloneNode(true));
    }

    // Sort sections: documents with missing images first
    documentSections.sort((a, b) => {
        const aMissing = hasMissingImages(a);
        const bMissing = hasMissingImages(b);

        if (aMissing && !bMissing) return -1;
        if (!aMissing && bMissing) return 1;
        return 0;
    });

    // Remove all document sections
    documentSections.forEach(section => section.remove());

    // Re-add them in sorted order
    const lastElement = container.querySelector('.progress-info');
    documentSections.forEach(section => {
        container.insertBefore(section, lastElement.nextSibling);
    });

    // Update button states
    document.getElementById('sort-btn').style.display = 'none';
    document.getElementById('reset-sort-btn').style.display = 'inline-block';

    isSorted = true;

    // Count documents with missing images
    const missingCount = documentSections.filter(section => hasMissingImages(section)).length;
    showNotification(`Sorted! ${missingCount} documents with missing images moved to top.`, 'success');
}

function resetSort() {
    if (originalOrder.length === 0) return;

    const container = document.querySelector('.container');
    const documentSections = Array.from(container.querySelectorAll('.document-section'));

    // Remove all document sections
    documentSections.forEach(section => section.remove());

    // Re-add them in original order
    const lastElement = container.querySelector('.progress-info');
    originalOrder.forEach(section => {
        container.insertBefore(section.cloneNode(true), lastElement.nextSibling);
    });

    // Update button states
    document.getElementById('sort-btn').style.display = 'inline-block';
    document.getElementById('reset-sort-btn').style.display = 'none';

    isSorted = false;

    showNotification('Documents restored to original order.', 'success');

    // Re-initialize event handlers for the restored elements
    setupEventHandlers();
}

function setupEventHandlers() {
    // Re-setup tooltips for new elements
    const saveButtons = document.querySelectorAll('.save-btn');
    saveButtons.forEach(btn => {
        btn.title = 'Save changes (Ctrl+S)';
    });

    const resetButtons = document.querySelectorAll('.reset-btn');
    resetButtons.forEach(btn => {
        btn.title = 'Reset to original values (Esc)';
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    console.log('Image Editor initialized');

    // Setup optional features
    setupAutoSave();
    setupKeyboardShortcuts();
    setupBulkOperations();
    setupEventHandlers();

    showNotification('Image Editor ready! Use Ctrl+S to save, Esc to reset.', 'success');
});
