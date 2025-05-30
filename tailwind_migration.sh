#!/bin/bash

# Tailwind CSS CLI Migration Script for Ovarian Cancer Detection System
# This script migrates from CDN to CLI for better performance and customization

echo "🚀 Starting Tailwind CSS CLI Migration..."

# Step 1: Install Tailwind CSS CLI
echo "📦 Installing Tailwind CSS CLI..."
npm init -y
npm install -D tailwindcss
npx tailwindcss init

echo "✅ Tailwind CSS CLI installed successfully!"

# Step 2: Create tailwind.config.js with your custom colors
echo "⚙️ Creating tailwind.config.js with custom configuration..."
cat > tailwind.config.js << 'EOF'
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/js/**/*.js",
    "./app/**/*.py",
    // Add any other files that contain Tailwind classes
  ],
  theme: {
    extend: {
      colors: {
        primary: '#3B82F6',
        secondary: '#8B5CF6',
        accent: '#10B981',
        danger: '#EF4444',
        warning: '#F59E0B',
        dark: '#1F2937',
        // CAMLAS brand colors
        'camlas-red': '#D63031',
        'camlas-dark': '#2D3436',
        'camlas-gray': '#636E72',
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'float-delayed': 'float 6s ease-in-out infinite 2s',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-up': 'slideUp 0.8s ease-out',
        'smooth-slide-up': 'smoothSlideUp 1s ease-out',
        'bounce-gentle': 'bounceGentle 2s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(30px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        smoothSlideUp: {
          '0%': { transform: 'translateY(50px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        bounceGentle: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-5px)' },
        },
      },
      fontFamily: {
        'sans': ['Helvetica', 'Arial', 'sans-serif'],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      maxWidth: {
        '8xl': '88rem',
        '9xl': '96rem',
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [
    // Add any plugins you might need
    // require('@tailwindcss/forms'),
    // require('@tailwindcss/typography'),
  ],
}
EOF

# Step 3: Create input CSS file
echo "🎨 Creating input CSS file..."
mkdir -p static/css/src
cat > static/css/src/input.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Custom component classes */
@layer components {
  .nav-link {
    @apply text-gray-700 hover:text-primary px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200;
  }

  .nav-link.active {
    @apply text-primary bg-blue-50 font-semibold;
  }

  .mobile-nav-link {
    @apply text-gray-700 hover:text-primary hover:bg-gray-50 block px-3 py-2 rounded-md text-base font-medium transition-colors duration-200;
  }

  .mobile-nav-link.active {
    @apply text-primary bg-blue-50 font-semibold;
  }

  .citation-tab {
    @apply px-6 py-2 rounded-lg font-semibold text-sm transition-all duration-200 text-gray-600 hover:text-gray-800;
  }

  .citation-tab.active {
    @apply bg-blue-600 text-white;
  }

  .citation-content {
    @apply block;
  }

  .citation-content.hidden {
    @apply hidden;
  }

  .workflow-placeholder {
    @apply text-center py-12 text-gray-400;
  }

  .loading-dots::after {
    content: '';
    animation: dots 1.5s steps(4, end) infinite;
  }
}

/* Custom utility classes */
@layer utilities {
  .text-shadow {
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
  }

  .bg-gradient-primary {
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
  }

  .bg-gradient-secondary {
    background: linear-gradient(135deg, #8B5CF6 0%, #10B981 100%);
  }

  .scroll-smooth {
    scroll-behavior: smooth;
  }
}

/* Keyframe animations */
@keyframes dots {
  0%, 20% { content: '.'; }
  40% { content: '..'; }
  60% { content: '...'; }
  80%, 100% { content: ''; }
}

/* Scroll to top button styles */
#scrollToTop {
  @apply fixed bottom-6 right-6 w-12 h-12 bg-primary text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-110 z-50 opacity-0 pointer-events-none;
}

#scrollToTop.show {
  @apply opacity-100 pointer-events-auto;
}

/* Loading overlay styles */
.loading-dots::after {
  animation: dots 1.5s steps(4, end) infinite;
}

/* Responsive utilities */
@media (max-width: 640px) {
  .hero-title {
    @apply text-4xl;
  }
}

@media (min-width: 768px) {
  .hero-title {
    @apply text-6xl;
  }
}
EOF

# Step 4: Create build script
echo "🔧 Creating build script..."
mkdir -p scripts
cat > scripts/build-css.sh << 'EOF'
#!/bin/bash
echo "🎨 Building Tailwind CSS..."
npx tailwindcss -i ./static/css/src/input.css -o ./static/css/output.css --watch
EOF

chmod +x scripts/build-css.sh

# Step 5: Create package.json scripts
echo "📝 Adding npm scripts..."
cat > package.json << 'EOF'
{
  "name": "ovarian-cancer-detection-system",
  "version": "1.0.0",
  "description": "AI-powered ovarian cancer detection using deep learning",
  "scripts": {
    "build-css": "tailwindcss -i ./static/css/src/input.css -o ./static/css/output.css",
    "watch-css": "tailwindcss -i ./static/css/src/input.css -o ./static/css/output.css --watch",
    "build-css-prod": "tailwindcss -i ./static/css/src/input.css -o ./static/css/output.css --minify"
  },
  "devDependencies": {
    "tailwindcss": "^3.4.0"
  },
  "keywords": [
    "ovarian-cancer",
    "ai",
    "deep-learning",
    "medical-imaging",
    "camlas"
  ],
  "author": "CAMLAS Innovation Hub",
  "license": "MIT"
}
EOF

# Step 6: Build initial CSS
echo "🏗️ Building initial CSS..."
npm run build-css

echo "✅ Tailwind CSS CLI migration completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Update your base.html template to use the new CSS file"
echo "2. Remove the Tailwind CDN script tag"
echo "3. Test your application to ensure styles are working"
echo ""
echo "🚀 Development Commands:"
echo "- npm run watch-css    # Watch for changes during development"
echo "- npm run build-css    # Build CSS for development"
echo "- npm run build-css-prod # Build minified CSS for production"