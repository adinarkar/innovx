/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // innovX palette - pastel red + white + dark grey (build spec section 2)
        brand: {
          DEFAULT: '#E57373',
          light: '#F6CACA',
          bg: '#FFF5F5',
          dark: '#D26060',
          deep: '#B94F4F',
        },
        ink: {
          DEFAULT: '#2B2B2B',
          soft: '#454545',
          muted: '#777777',
          line: '#E5E5E5',
        },
        state: {
          ok: '#4F9D69',
          warn: '#C9973F',
          bad: '#B94F4F',
        },
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'SFMono-Regular', 'Consolas', 'monospace'],
      },
      boxShadow: {
        card: '0 1px 2px rgba(43,43,43,0.04), 0 8px 24px rgba(43,43,43,0.06)',
        lift: '0 2px 4px rgba(43,43,43,0.06), 0 16px 40px rgba(43,43,43,0.10)',
      },
      borderRadius: { xl2: '1.125rem' },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        sweep: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(300%)' },
        },
        'pulse-soft': {
          '0%,100%': { opacity: '1' },
          '50%': { opacity: '0.45' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.35s ease-out both',
        sweep: 'sweep 1.4s ease-in-out infinite',
        'pulse-soft': 'pulse-soft 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
