<!--
  components/content/Mermaid.vue
  Nuxt Content MDC component — rendered whenever ::mermaid blocks appear in .md files.
  Usage in markdown:
    ::mermaid
    flowchart LR
      A --> B
    ::
-->
<template>
  <ClientOnly>
    <MermaidDiagram :code="code" />
    <template #fallback>
      <div class="mermaid-diagram animate-pulse h-32 bg-gray-800 rounded-xl" />
    </template>
  </ClientOnly>
</template>

<script setup lang="ts">
defineProps<{ code?: string }>()
</script>
