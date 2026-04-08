// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  devtools: { enabled: false },

  modules: [
    '@nuxt/content',
    '@nuxtjs/tailwindcss',
  ],

  content: {
    highlight: {
      theme: 'github-dark',
      langs: [
        'bash',
        'json',
        'yaml',
        'typescript',
        'javascript',
        'python',
      ],
    },
    navigation: {
      fields: ['title', 'description', 'icon'],
    },
  },

  app: {
    baseURL: process.env.NUXT_APP_BASE_URL || '/',
    head: {
      title: 'HoloLang Documentation',
      meta: [
        { name: 'description', content: 'Official documentation for HoloLang — a custom DSL for holographic device control, tensor processing, and light-manipulation automation.' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      ],
      link: [
        { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' },
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap' },
      ],
    },
  },

  nitro: {
    preset: 'node-server',
  },

  compatibilityDate: '2024-11-01',
})
