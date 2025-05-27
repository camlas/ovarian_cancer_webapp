/* =================================
   SIMPLE ANIMATION CONTROLLER
   Extracted from base.html
   ================================= */

class SimpleAnimationController {
    constructor() {
        this.observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        this.init();
    }

    init() {
        this.setupIntersectionObserver();
        this.addKeyframes();
        console.log('🎨 Simple Animation Controller initialized');
    }

    // Setup observer to watch for elements entering viewport
    setupIntersectionObserver() {
        this.observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateElement(entry.target);
                }
            });
        }, this.observerOptions);

        // Start observing elements
        this.observeElements();
    }

    // Find and observe elements that need animation
    observeElements() {
        // Elements with animation classes
        const selectors = [
            '.animate-fade-in',
            '.animate-slide-up',
            '.animate-slide-down',
            '.animate-slide-left',
            '.animate-slide-right',
            '.animate-scale-up'
        ];

        selectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                if (!el.hasAttribute('data-animated')) {
                    // Hide element initially
                    el.style.opacity = '0';
                    el.style.transform = this.getInitialTransform(selector);
                    el.style.transition = 'all 0.6s ease-out';

                    this.observer.observe(el);
                }
            });
        });
    }

    // Get initial transform based on animation type
    getInitialTransform(selector) {
        switch (selector) {
            case '.animate-slide-up':
                return 'translateY(30px)';
            case '.animate-slide-down':
                return 'translateY(-30px)';
            case '.animate-slide-left':
                return 'translateX(30px)';
            case '.animate-slide-right':
                return 'translateX(-30px)';
            case '.animate-scale-up':
                return 'scale(0.8)';
            default:
                return 'translateY(0)';
        }
    }

    // Animate element when it appears
    animateElement(element) {
        if (element.hasAttribute('data-animated')) return;

        element.setAttribute('data-animated', 'true');

        // Add delay if specified
        const delay = element.getAttribute('data-delay') || 0;

        setTimeout(() => {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0) translateX(0) scale(1)';
        }, parseInt(delay));
    }

    // Manual animation methods
    fadeIn(element, duration = 600) {
        element.style.opacity = '0';
        element.style.transition = `opacity ${duration}ms ease-out`;

        setTimeout(() => {
            element.style.opacity = '1';
        }, 10);
    }

    slideUp(element, duration = 600) {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        element.style.transition = `all ${duration}ms ease-out`;

        setTimeout(() => {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, 10);
    }

    scaleUp(element, duration = 600) {
        element.style.opacity = '0';
        element.style.transform = 'scale(0.8)';
        element.style.transition = `all ${duration}ms ease-out`;

        setTimeout(() => {
            element.style.opacity = '1';
            element.style.transform = 'scale(1)';
        }, 10);
    }

    // Animate multiple elements with stagger
    animateGroup(elements, animationType = 'slideUp', staggerDelay = 100) {
        elements.forEach((element, index) => {
            setTimeout(() => {
                switch (animationType) {
                    case 'fadeIn':
                        this.fadeIn(element);
                        break;
                    case 'slideUp':
                        this.slideUp(element);
                        break;
                    case 'scaleUp':
                        this.scaleUp(element);
                        break;
                }
            }, index * staggerDelay);
        });
    }

    // Show element with animation
    show(element, animationType = 'fadeIn') {
        element.classList.remove('hidden');

        switch (animationType) {
            case 'fadeIn':
                this.fadeIn(element);
                break;
            case 'slideUp':
                this.slideUp(element);
                break;
            case 'scaleUp':
                this.scaleUp(element);
                break;
        }
    }

    // Hide element with animation
    hide(element, animationType = 'fadeOut') {
        const duration = 300;

        switch (animationType) {
            case 'fadeOut':
                element.style.transition = `opacity ${duration}ms ease-out`;
                element.style.opacity = '0';
                break;
            case 'slideDown':
                element.style.transition = `all ${duration}ms ease-out`;
                element.style.opacity = '0';
                element.style.transform = 'translateY(30px)';
                break;
            case 'scaleDown':
                element.style.transition = `all ${duration}ms ease-out`;
                element.style.opacity = '0';
                element.style.transform = 'scale(0.8)';
                break;
        }

        setTimeout(() => {
            element.classList.add('hidden');
        }, duration);
    }

    // Re-observe new elements (call when DOM changes)
    refresh() {
        this.observeElements();
    }

    // Cleanup
    destroy() {
        if (this.observer) {
            this.observer.disconnect();
        }
    }

    // Add CSS keyframes and utility classes
    addKeyframes() {
        if (!document.getElementById('simple-animations-css')) {
            const style = document.createElement('style');
            style.id = 'simple-animations-css';
            style.textContent = `
                /* Additional animation utility classes */
                .animate-fade-in-delayed {
                    animation: fadeIn 0.6s ease-out forwards;
                    animation-delay: 0.3s;
                    opacity: 0;
                }
                
                .animate-slide-up-delayed {
                    animation: slideUp 0.6s ease-out forwards;
                    animation-delay: 0.3s;
                    opacity: 0;
                }
                
                .animate-scale-up-delayed {
                    animation: scaleUp 0.6s ease-out forwards;
                    animation-delay: 0.3s;
                    opacity: 0;
                }
            `;
            document.head.appendChild(style);
        }
    }
}

// Initialize on DOM ready
let simpleAnimations;

document.addEventListener('DOMContentLoaded', () => {
    simpleAnimations = new SimpleAnimationController();
    window.simpleAnimations = simpleAnimations;
});

// Utility functions for easy use
function animateIn(element, type = 'fadeIn') {
    if (simpleAnimations) {
        simpleAnimations.show(element, type);
    }
}

function animateOut(element, type = 'fadeOut') {
    if (simpleAnimations) {
        simpleAnimations.hide(element, type);
    }
}

function animateGroup(elements, type = 'slideUp', delay = 100) {
    if (simpleAnimations) {
        simpleAnimations.animateGroup(elements, type, delay);
    }
}

// Export for module use if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SimpleAnimationController;
}