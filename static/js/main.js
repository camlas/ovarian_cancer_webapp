/* =================================
   OVARIAN CANCER AI DETECTION SYSTEM
   ================================= */

/* =============================================================================
   1. INITIALIZATION & DOM READY
   ============================================================================= */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components with error handling
    try {
        initializeScrollToTop();
        initializeNavigation();
        initializeUpload();
    } catch (error) {
        console.error('❌ Initialization error:', error);
    }
});

/* =============================================================================
   2. SCROLL TO TOP FUNCTIONALITY - FIXED
   ============================================================================= */

function initializeScrollToTop() {
    const scrollToTopBtn = document.getElementById('scrollToTop');

    if (!scrollToTopBtn) {
        return;
    }

    // FIXED: Better scroll detection
    function toggleScrollButton() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

        if (scrollTop > 300) {
            scrollToTopBtn.classList.add('show');
        } else {
            scrollToTopBtn.classList.remove('show');
        }
    }

    // FIXED: Use throttled scroll event for better performance
    let ticking = false;
    function handleScroll() {
        if (!ticking) {
            requestAnimationFrame(() => {
                toggleScrollButton();
                ticking = false;
            });
            ticking = true;
        }
    }

    window.addEventListener('scroll', handleScroll);

    // Initial check
    toggleScrollButton();

    // Smooth scroll to top with easing
    scrollToTopBtn.addEventListener('click', function(e) {
        e.preventDefault();
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Enhanced hover effects
    scrollToTopBtn.addEventListener('mouseenter', function() {
        const icon = this.querySelector('i');
        if (icon) icon.classList.add('animate-bounce');
    });

    scrollToTopBtn.addEventListener('mouseleave', function() {
        const icon = this.querySelector('i');
        if (icon) icon.classList.remove('animate-bounce');
    });

}

/* =============================================================================
   3. NAVIGATION FUNCTIONALITY
   ============================================================================= */

function initializeNavigation() {
    // Mobile menu toggle
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');

    if (mobileMenuButton && mobileMenu) {
        mobileMenuButton.addEventListener('click', function() {
            mobileMenu.classList.toggle('hidden');

            // Update icon
            const icon = this.querySelector('i');
            if (icon) {
                if (mobileMenu.classList.contains('hidden')) {
                    icon.className = 'fas fa-bars text-xl';
                } else {
                    icon.className = 'fas fa-times text-xl';
                }
            }
        });
    }

    // Smooth scroll for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/* =============================================================================
   4. FILE UPLOAD FUNCTIONALITY
   ============================================================================= */

function initializeUpload() {
    setupUpload();
}

function setupUpload() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');

    if (!uploadArea || !fileInput) {
        return;
    }

    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    // File selected
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            processFile(file);
        }
    });

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.borderColor = '#3B82F6';
        this.style.backgroundColor = '#EBF8FF';
        this.style.transform = 'scale(1.02)';
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.style.borderColor = '';
        this.style.backgroundColor = '';
        this.style.transform = 'scale(1)';
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.borderColor = '';
        this.style.backgroundColor = '';
        this.style.transform = 'scale(1)';

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            processFile(files[0]);
        }
    });
}

async function processFile(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        showAlert('Please select an image file', 'error');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showAlert('File too large (max 10MB)', 'error');
        return;
    }

    showLoading(true);
    showProgress(true);

    try {
        // Upload file
        updateProgress(20);
        const formData = new FormData();
        formData.append('file', file);

        const uploadResponse = await fetch('/test/upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) throw new Error('Upload failed');
        const uploadResult = await uploadResponse.json();

        updateProgress(50);

        // Preview image
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.getElementById('preview-image');
            const name = document.getElementById('image-filename');
            if (img) img.src = e.target.result;
            if (name) name.textContent = file.name;
        };
        reader.readAsDataURL(file);

        updateProgress(70);

        // Make prediction
        const predFormData = new FormData();
        predFormData.append('file_path', uploadResult.file_path);

        const predResponse = await fetch('/test/predict', {
            method: 'POST',
            body: predFormData
        });

        if (!predResponse.ok) throw new Error('Prediction failed');
        const result = await predResponse.json();

        updateProgress(100);

        // Show results with delay for smooth UX
        setTimeout(() => {
            showResults(result);
            showAlert('Analysis completed successfully!', 'success');
        }, 500);

    } catch (error) {
        showAlert('Error: ' + error.message, 'error');
    } finally {
        showLoading(false);
        setTimeout(() => showProgress(false), 1000);
    }
}

/* =============================================================================
   5. RESULTS DISPLAY FUNCTIONS
   ============================================================================= */

function showResults(result) {
    // Hide upload, show results with animation
    const uploadContainer = document.getElementById('upload-area').parentElement;
    const resultsSection = document.getElementById('results-section');

    if (uploadContainer) uploadContainer.classList.add('hidden');
    if (resultsSection) resultsSection.classList.remove('hidden');

    // Update prediction with enhanced styling
    const predText = document.getElementById('prediction-text');
    if (predText) {
        predText.textContent = result.prediction;
        predText.className = result.prediction === 'Cancer'
            ? 'text-4xl font-bold mb-2 text-red-600'
            : 'text-4xl font-bold mb-2 text-green-600';
    }

    // Update probability
    const probText = document.getElementById('probability-text');
    if (probText) {
        probText.textContent = `Probability: ${(result.probability * 100).toFixed(1)}%`;
    }

    // Update confidence with animated progress bar
    updateConfidenceDisplay(result.confidence);

    // Update technical details
    updateTechnicalDetails(result);

    // Update prediction card styling
    updatePredictionCard(result);

    // Smooth scroll to results
    setTimeout(() => {
        if (resultsSection) {
            resultsSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    }, 100);

    // Store result globally for report generation
    window.currentPredictionResult = result;
}

function updateConfidenceDisplay(confidence) {
    const confText = document.getElementById('confidence-text');
    const confBar = document.getElementById('confidence-bar');

    if (!confText || !confBar) return;

    const confidencePercent = (confidence * 100).toFixed(1);
    confText.textContent = `${confidencePercent}%`;

    // Animate confidence bar
    confBar.style.width = '0%';
    setTimeout(() => {
        confBar.style.width = `${confidencePercent}%`;
    }, 300);

    // Set color based on confidence level
    let barClass = 'h-3 rounded-full transition-all duration-1000 ';
    if (confidence >= 0.8) {
        barClass += 'bg-gradient-to-r from-green-400 to-green-600';
    } else if (confidence >= 0.6) {
        barClass += 'bg-gradient-to-r from-yellow-400 to-yellow-600';
    } else {
        barClass += 'bg-gradient-to-r from-red-400 to-red-600';
    }
    confBar.className = barClass;
}

function updateTechnicalDetails(result) {
    // Processing time
    const procTime = document.getElementById('processing-time');
    if (procTime) {
        const time = result.processing_time ? `${result.processing_time.toFixed(2)}s` : 'N/A';
        procTime.textContent = time;
    }

    // Device info
    const device = document.getElementById('device-info');
    if (device) {
        const deviceName = result.model_info?.device || 'Unknown';
        device.textContent = deviceName.toUpperCase();
    }

    // Features count
    const features = document.getElementById('features-count');
    if (features) {
        const featCount = result.model_info?.selected_features_count || 'N/A';
        features.textContent = featCount;
    }

    // Feature extractor
    const featureExtractor = document.getElementById('feature-extractor');
    if (featureExtractor) {
        const extractorName = result.model_info?.feature_extractor || 'N/A';
        featureExtractor.textContent = extractorName;
    }

    // Classifier
    const classifier = document.getElementById('classifier');
    if (classifier) {
        const classifierName = result.model_info?.classifier || 'N/A';
        classifier.textContent = classifierName;
    }
}

function updatePredictionCard(result) {
    const card = document.getElementById('prediction-card');
    if (!card) return;

    if (result.prediction === 'Cancer') {
        card.className = 'text-center p-6 rounded-xl bg-gradient-to-br from-red-50 to-red-100 border-2 border-red-200 shadow-inner';
    } else {
        card.className = 'text-center p-6 rounded-xl bg-gradient-to-br from-green-50 to-green-100 border-2 border-green-200 shadow-inner';
    }
}

/* =============================================================================
   6. PROGRESS & LOADING FUNCTIONS
   ============================================================================= */

function showProgress(show) {
    const progress = document.getElementById('upload-progress');
    if (progress) {
        if (show) {
            progress.classList.remove('hidden');
        } else {
            progress.classList.add('hidden');
            const progressBar = document.getElementById('progress-bar');
            if (progressBar) progressBar.style.width = '0%';
        }
    }
}

function updateProgress(percent) {
    const bar = document.getElementById('progress-bar');
    if (bar) {
        bar.style.width = `${percent}%`;
    }
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

/* =============================================================================
   7. ALERT/NOTIFICATION SYSTEM
   ============================================================================= */

function showAlert(message, type = 'info') {
    const colors = {
        success: 'bg-gradient-to-r from-green-500 to-green-600',
        error: 'bg-gradient-to-r from-red-500 to-red-600',
        warning: 'bg-gradient-to-r from-yellow-500 to-yellow-600',
        info: 'bg-gradient-to-r from-blue-500 to-blue-600'
    };

    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-triangle',
        warning: 'fas fa-exclamation-circle',
        info: 'fas fa-info-circle'
    };

    const alert = document.createElement('div');
    alert.className = `fixed top-4 right-4 z-50 ${colors[type]} text-white p-4 rounded-lg shadow-lg max-w-sm transform translate-x-full transition-transform duration-300`;
    alert.innerHTML = `
        <div class="flex items-center">
            <i class="${icons[type]} mr-3"></i>
            <span class="flex-1">${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-3 text-white hover:text-gray-200">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

    document.body.appendChild(alert);

    // Animate in
    setTimeout(() => {
        alert.style.transform = 'translateX(0)';
    }, 100);

    // Auto remove after 4 seconds
    setTimeout(() => {
        alert.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (alert.parentElement) alert.remove();
        }, 300);
    }, 4000);
}

/* =============================================================================
   8. SAMPLE IMAGES FUNCTIONALITY
   ============================================================================= */

window.showSampleModal = function() {
    const modal = document.getElementById('sample-modal');
    if (modal) {
        modal.classList.remove('hidden');
        loadSampleImages();
    }
};

window.closeSampleModal = function() {
    const modal = document.getElementById('sample-modal');
    if (modal) {
        modal.classList.add('hidden');
    }
};

async function loadSampleImages() {
    try {
        const response = await fetch('/test/sample-images');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        const container = document.getElementById('sample-images-container');
        if (!container) return;

        container.innerHTML = '';
        container.className = 'w-full grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6 p-4';

        // Add cancer samples
        if (data.cancer_samples && data.cancer_samples.length > 0) {
            data.cancer_samples.forEach((path, i) => {
                const card = createSampleCard(path, i + 1, 'cancer');
                container.appendChild(card);
            });
        }

        // Add normal samples
        if (data.non_cancer_samples && data.non_cancer_samples.length > 0) {
            data.non_cancer_samples.forEach((path, i) => {
                const card = createSampleCard(path, i + 1, 'normal');
                container.appendChild(card);
            });
        }

    } catch (error) {
        const container = document.getElementById('sample-images-container');
        if (container) {
            container.innerHTML = '<div class="col-span-full text-center py-8"><p class="text-red-500"><i class="fas fa-exclamation-triangle mr-2"></i>Error loading sample images</p></div>';
        }
    }
}

function createSampleCard(path, index, type) {
    const isCancer = type === 'cancer';
    const imageUrl = `/${path}`;

    const div = document.createElement('div');
    div.className = 'sample-image-card transform hover:scale-105 transition-all duration-300 w-full';

    const cardDiv = document.createElement('div');
    cardDiv.className = isCancer
        ? 'bg-gradient-to-br from-red-50 to-red-100 p-4 rounded-xl cursor-pointer hover:from-red-100 hover:to-red-200 transition-all duration-300 border border-red-200 shadow-lg hover:shadow-xl w-full'
        : 'bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-xl cursor-pointer hover:from-green-100 hover:to-green-200 transition-all duration-300 border border-green-200 shadow-lg hover:shadow-xl w-full';

    cardDiv.onclick = () => selectSample(path);

    const imageContainer = document.createElement('div');
    imageContainer.className = 'relative overflow-hidden rounded-lg mb-3 group aspect-square';

    const img = document.createElement('img');
    img.src = imageUrl;
    img.alt = `${type} Sample ${index}`;
    img.className = 'w-full h-full object-cover transition-transform duration-300 group-hover:scale-110';

    img.onerror = function() {
        const fallbackIcon = isCancer
            ? '<i class="fas fa-microscope text-red-500 text-3xl"></i>'
            : '<i class="fas fa-check-circle text-green-500 text-3xl"></i>';
        const fallbackBg = isCancer ? 'bg-red-200' : 'bg-green-200';
        this.parentElement.innerHTML = `<div class="w-full h-full ${fallbackBg} flex items-center justify-center rounded-lg">${fallbackIcon}</div>`;
    };

    const overlay = document.createElement('div');
    overlay.className = 'absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end justify-center pb-2';
    overlay.innerHTML = '<span class="text-white text-xs font-semibold bg-black/50 px-2 py-1 rounded">Click to Analyze</span>';

    const badge = document.createElement('div');
    badge.className = isCancer
        ? 'absolute top-2 right-2 bg-red-500 text-white text-xs px-2 py-1 rounded-full font-bold shadow-md'
        : 'absolute top-2 right-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full font-bold shadow-md';
    badge.textContent = isCancer ? 'CANCER' : 'NORMAL';

    const textContainer = document.createElement('div');
    textContainer.className = 'text-center';
    textContainer.innerHTML = `
        <h4 class="text-sm font-bold ${isCancer ? 'text-red-800' : 'text-green-800'} mb-1">
            ${isCancer ? 'Cancer' : 'Normal'} Sample ${index}
        </h4>
    `;

    imageContainer.appendChild(img);
    imageContainer.appendChild(overlay);
    imageContainer.appendChild(badge);
    cardDiv.appendChild(imageContainer);
    cardDiv.appendChild(textContainer);
    div.appendChild(cardDiv);

    return div;
}

window.selectSample = async function(imagePath) {
    closeSampleModal();
    showLoading(true);

    try {
        const formData = new FormData();
        formData.append('sample_path', imagePath);

        const response = await fetch('/test/predict-sample', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        // Set preview for sample
        const img = document.getElementById('preview-image');
        const name = document.getElementById('image-filename');

        const imageUrl = `/${imagePath}`;
        if (img) {
            img.src = imageUrl;
            img.onerror = function() {
                this.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="150"><rect width="200" height="150" fill="%23f3f4f6"/><text x="100" y="75" text-anchor="middle" dy=".3em" fill="%236b7280">Sample Image</text></svg>';
            };
        }

        if (name) {
            name.textContent = imagePath.split('/').pop();
        }

        showResults(result);
        showAlert('Sample analysis completed!', 'success');

    } catch (error) {
        showAlert('Sample test failed: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
};

/* =============================================================================
   9. REPORT GENERATION
   ============================================================================= */

window.generateReport = async function() {
    if (!window.currentPredictionResult) {
        showAlert('No prediction result available for report generation.', 'warning');
        return;
    }

    try {
        showAlert('Generating PDF report...', 'info');

        const response = await fetch('/report/quick-generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prediction_result: window.currentPredictionResult
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
            showAlert('Report generated successfully!', 'success');

            // Download the report
            const downloadUrl = result.download_url;
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = result.filename || 'ovarian_cancer_report.pdf';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

        } else {
            throw new Error(result.message || 'Report generation failed');
        }

    } catch (error) {
        showAlert('Failed to generate report: ' + error.message, 'error');
    }
};

/* =============================================================================
   10. UTILITY FUNCTIONS
   ============================================================================= */

window.resetUpload = function() {
    const uploadContainer = document.getElementById('upload-area')?.parentElement;
    const resultsSection = document.getElementById('results-section');
    const fileInput = document.getElementById('file-input');

    if (uploadContainer) uploadContainer.classList.remove('hidden');
    if (resultsSection) resultsSection.classList.add('hidden');
    if (fileInput) fileInput.value = '';

    window.currentPredictionResult = null;
    showAlert('Ready for new upload!', 'info');
};

window.scrollToUpload = function() {
    const uploadSection = document.getElementById('upload-section');
    if (uploadSection) {
        uploadSection.scrollIntoView({ behavior: 'smooth' });
    }
};

/* =============================================================================
   11. GLOBAL ANIMATION UTILITIES
   ============================================================================= */

// Global animation utilities for external use
window.animateElement = function(element, animationType = 'fadeIn', delay = 0) {
    if (!element) return;

    setTimeout(() => {
        element.classList.add(`animate-${animationType}`);
    }, delay);
};

window.showWithAnimation = function(element, animationType = 'fade-in') {
    if (!element) return;

    element.classList.remove('hidden');
    element.classList.add(`animate-${animationType}`);
};

window.hideWithAnimation = function(element, animationType = 'fade-out') {
    if (!element) return;

    element.classList.add('opacity-0');
    setTimeout(() => {
        element.classList.add('hidden');
    }, 300);
};

/* =============================================================================
   12. ERROR HANDLING & DEBUGGING
   ============================================================================= */

// Global error handler
window.addEventListener('error', function(e) {
    showAlert('An unexpected error occurred. Please try again.', 'error');
});

// Unhandled promise rejection handler
window.addEventListener('unhandledrejection', function(e) {
    showAlert('A network error occurred. Please check your connection.', 'error');
});

/* =============================================================================
   13. PERFORMANCE MONITORING
   ============================================================================= */

// Simple performance monitoring
if (window.performance && window.performance.mark) {
    window.performance.mark('main-js-start');

    window.addEventListener('load', function() {
        window.performance.mark('main-js-end');
        try {
            window.performance.measure('main-js-load', 'main-js-start', 'main-js-end');
            const measure = window.performance.getEntriesByName('main-js-load')[0];
        } catch (error) {
            console.log('⚡ Performance measurement not available');
        }
    });
}

/* =============================================================================
   14. ALGORITHM WORKFLOW VIEW
   ============================================================================= */

function openImageModal(imageSrc) {
  const modal = document.createElement('div');
  modal.className = 'fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50';
  modal.style.animation = 'fadeIn 0.3s ease-in-out';

  const modalContent = `
    <div class="relative bg-white rounded-lg p-2 max-w-4xl w-full mx-auto shadow-2xl transform transition-all duration-300 scale-95 hover:scale-100">
      <img src="${imageSrc}" alt="Full size workflow" class="w-full h-auto rounded-lg" />
      <button onclick="this.closest('.fixed').remove()" class="absolute top-2 right-2 bg-black text-white rounded-full p-2 opacity-90 hover:opacity-100 transition-opacity">
        <i class="fas fa-times text-lg"></i>
      </button>
      <a href="${imageSrc}" download class="absolute bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm transition-colors flex items-center">
        <i class="fas fa-download mr-1"></i> Download
      </a>
    </div>
  `;

  modal.innerHTML = modalContent;
  document.body.appendChild(modal);
}