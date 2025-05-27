// Citation format switching and copying
function showCitationFormat(format) {
    // Hide all citation contents
    document.querySelectorAll('.citation-content').forEach(function(content) {
        content.classList.add('hidden');
    });

    // Remove active class from all tabs
    document.querySelectorAll('.citation-tab').forEach(function(tab) {
        tab.classList.remove('bg-blue-600', 'text-white');
        tab.classList.add('text-gray-600', 'hover:text-gray-800');
    });

    // Show selected citation content
    const selectedContent = document.getElementById(format + '-citation');
    if (selectedContent) {
        selectedContent.classList.remove('hidden');
    }

    // Activate selected tab
    const selectedTab = document.getElementById(format + '-tab');
    if (selectedTab) {
        selectedTab.classList.remove('text-gray-600', 'hover:text-gray-800');
        selectedTab.classList.add('bg-blue-600', 'text-white');
    }
}

// Copy citation to clipboard
function copyCitation(format) {
    const textElementId = format + '-text';
    const citationElement = document.getElementById(textElementId);

    if (!citationElement) {
        showCopyNotification('Citation not found', 'error');
        return;
    }

    const citationText = citationElement.innerText;
    const formatName = format === 'bibtex' ? 'BibTeX' : 'APA';
    copyTextToClipboard(citationText, formatName + ' citation copied to clipboard!');
}

// Generic copy function
function copyTextToClipboard(text, successMessage) {
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(function() {
            showCopyNotification(successMessage);
        }).catch(function(err) {
            console.error('Failed to copy citation: ', err);
            fallbackCopyTextToClipboard(text, successMessage);
        });
    } else {
        fallbackCopyTextToClipboard(text, successMessage);
    }
}

// Fallback copy function for older browsers
function fallbackCopyTextToClipboard(text, successMessage = 'Citation copied to clipboard!') {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.top = "0";
    textArea.style.left = "0";
    textArea.style.position = "fixed";

    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();

    try {
        const successful = document.execCommand('copy');
        if (successful) {
            showCopyNotification(successMessage);
        } else {
            showCopyNotification('Failed to copy citation', 'error');
        }
    } catch (err) {
        console.error('Fallback: Oops, unable to copy', err);
        showCopyNotification('Failed to copy citation', 'error');
    }

    document.body.removeChild(textArea);
}

// Show copy notification
function showCopyNotification(message, type = 'success') {
    const notification = document.createElement('div');
    const bgColor = type === 'success' ? 'from-green-500 to-green-600' : 'from-red-500 to-red-600';
    const icon = type === 'success' ? 'fa-check-circle' : 'fa-exclamation-triangle';

    notification.className = `fixed top-4 right-4 z-50 bg-gradient-to-r ${bgColor} text-white p-4 rounded-lg shadow-lg max-w-sm transform translate-x-full transition-transform duration-300`;
    notification.innerHTML = `
        <div class="flex items-center">
            <i class="fas ${icon} mr-3"></i>
            <span>${message}</span>
        </div>
    `;

    document.body.appendChild(notification);

    // Animate in
    setTimeout(function() {
        notification.style.transform = 'translateX(0)';
    }, 100);

    // Auto remove after 3 seconds
    setTimeout(function() {
        notification.style.transform = 'translateX(100%)';
        setTimeout(function() {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 300);
    }, 3000);
}

// Add smooth scroll for any anchor links
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
        anchor.addEventListener('click', function(e) {
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
});