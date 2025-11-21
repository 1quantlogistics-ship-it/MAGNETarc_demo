/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          900: '#0a0e1a',
          800: '#1a1a2e',
          700: '#16213e',
          600: '#0f3460',
          500: '#1a508b',
        },
        cyan: {
          DEFAULT: '#00d9ff',
          light: '#5ce1e6',
          dark: '#00a8cc',
        },
        success: '#00ff88',
        warning: '#ffd700',
        alert: '#ff4444',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite',
      },
      keyframes: {
        glow: {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.5 },
        },
      },
    },
  },
  plugins: [],
}
