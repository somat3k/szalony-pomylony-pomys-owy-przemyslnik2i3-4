<!--
  MermaidDiagram.vue
  Lazily renders a Mermaid diagram client-side.
  Usage:  <MermaidDiagram :code="diagramSource" />
-->
<template>
  <div class="mermaid-diagram">
    <div ref="container" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'

const props = defineProps<{ code: string }>()
const container = ref<HTMLElement | null>(null)

async function render() {
  if (!container.value || !props.code) return
  // Dynamic import keeps mermaid out of SSR bundle
  const mermaid = (await import('mermaid')).default
  mermaid.initialize({
    startOnLoad: false,
    theme: 'dark',
    themeVariables: {
      background: '#111827',
      primaryColor: '#2094a6',
      primaryTextColor: '#f3f4f6',
      primaryBorderColor: '#3bafc0',
      lineColor: '#6b7280',
      secondaryColor: '#1f2937',
      tertiaryColor: '#1f2937',
    },
  })
  const id = `mermaid-${Math.random().toString(36).slice(2)}`
  const { svg } = await mermaid.render(id, props.code)
  if (container.value) container.value.innerHTML = svg
}

onMounted(render)
watch(() => props.code, render)
</script>
