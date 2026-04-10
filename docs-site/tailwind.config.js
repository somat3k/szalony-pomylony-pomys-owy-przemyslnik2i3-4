/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './components/**/*.{js,vue,ts}',
    './layouts/**/*.vue',
    './pages/**/*.vue',
    './plugins/**/*.{js,ts}',
    './app.vue',
    './content/**/*.md',
  ],
  theme: {
    extend: {
      colors: {
        holo: {
          50:  '#f0fafb',
          100: '#d9f2f5',
          200: '#b2e5eb',
          300: '#7acfda',
          400: '#3bafc0',
          500: '#2094a6',
          600: '#1c758c',
          700: '#1c5f73',
          800: '#1d4f5f',
          900: '#1b4050',
          950: '#0d2835',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
