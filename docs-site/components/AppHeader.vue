<template>
  <header class="sticky top-0 z-50 bg-gray-950/90 backdrop-blur border-b border-gray-800 h-16 flex items-center px-6 gap-6">
    <!-- Logo -->
    <NuxtLink to="/" class="flex items-center gap-2 font-bold text-lg text-gray-100 hover:text-holo-400 transition-colors">
      <svg class="w-7 h-7 text-holo-400" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        <polygon points="16,2 30,10 30,22 16,30 2,22 2,10" stroke="currentColor" stroke-width="2" fill="none"/>
        <polygon points="16,8 24,13 24,19 16,24 8,19 8,13" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.15"/>
        <circle cx="16" cy="16" r="3" fill="currentColor"/>
      </svg>
      HoloLang
    </NuxtLink>

    <!-- Nav links -->
    <nav class="hidden md:flex items-center gap-4 text-sm text-gray-400">
      <NuxtLink to="/getting-started/introduction" class="hover:text-gray-100 transition-colors">Docs</NuxtLink>
      <NuxtLink to="/language/syntax" class="hover:text-gray-100 transition-colors">Language</NuxtLink>
      <NuxtLink to="/diagrams/architecture" class="hover:text-gray-100 transition-colors">Diagrams</NuxtLink>
      <NuxtLink to="/api-reference/cli" class="hover:text-gray-100 transition-colors">CLI</NuxtLink>
    </nav>

    <div class="flex-1" />

    <!-- Version badge -->
    <span class="hidden sm:inline-flex items-center gap-1 text-xs bg-holo-900 text-holo-300 px-2 py-1 rounded-full font-mono">
      v0.1.0
    </span>

    <!-- GitHub link -->
    <a
      href="https://github.com/somat3k/szalony-pomylony-pomys-owy-przemyslnik2i3-4"
      target="_blank"
      rel="noopener noreferrer"
      class="text-gray-400 hover:text-gray-100 transition-colors"
      aria-label="GitHub"
    >
      <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"/>
      </svg>
    </a>
  </header>
</template>
