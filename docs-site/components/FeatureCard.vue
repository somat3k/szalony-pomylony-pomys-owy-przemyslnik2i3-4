<template>
  <div class="bg-gray-800 border border-gray-700 rounded-xl p-5 hover:border-holo-700 transition-colors">
    <div class="text-2xl mb-3">{{ icon }}</div>
    <h3 class="font-semibold text-gray-100 mb-1">{{ title }}</h3>
    <p class="text-sm text-gray-400">{{ desc }}</p>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  icon: string
  title: string
  desc: string
}>()
</script>
