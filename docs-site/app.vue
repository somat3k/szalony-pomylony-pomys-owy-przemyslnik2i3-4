<template>
  <div class="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
    <AppHeader />
    <div class="flex flex-1 max-w-screen-2xl mx-auto w-full">
      <AppSidebar />
      <main class="flex-1 min-w-0 px-6 py-8">
        <NuxtPage />
      </main>
    </div>
    <AppFooter />
  </div>
</template>

<script setup lang="ts">
useHead({
  htmlAttrs: { lang: 'en' },
})
</script>

<style>
/* Global prose overrides for dark theme */
.prose pre {
  @apply bg-gray-900 border border-gray-700 rounded-lg;
}
.prose code:not(pre code) {
  @apply bg-gray-800 text-holo-300 px-1.5 py-0.5 rounded text-sm;
}
.prose h1, .prose h2, .prose h3 {
  @apply text-gray-100;
}
.prose p, .prose li {
  @apply text-gray-300;
}
.prose a {
  @apply text-holo-400 hover:text-holo-300;
}
.prose blockquote {
  @apply border-l-holo-500 text-gray-400;
}
.prose table th {
  @apply bg-gray-800 text-gray-200;
}
.prose table td {
  @apply border-gray-700 text-gray-300;
}
/* Mermaid diagram container */
.mermaid-diagram {
  @apply my-6 p-4 bg-gray-900 rounded-xl border border-gray-700 overflow-x-auto;
}
</style>
