<!--
  Catch-all docs page — renders any content/*.md file.
  URL: /getting-started/introduction → content/1.getting-started/1.introduction.md
-->
<template>
  <article>
    <ContentDoc v-slot="{ doc }">
      <div class="prose prose-invert prose-lg max-w-none">
        <h1 class="text-gray-100">{{ doc.title }}</h1>
        <p v-if="doc.description" class="lead text-gray-400">{{ doc.description }}</p>
        <ContentRenderer :value="doc" />
      </div>

      <!-- Prev / next navigation -->
      <nav class="mt-12 pt-8 border-t border-gray-800 flex justify-between gap-4">
        <NuxtLink
          v-if="prev"
          :to="prev._path"
          class="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-100 transition-colors"
        >
          ← {{ prev.title }}
        </NuxtLink>
        <span v-else />
        <NuxtLink
          v-if="next"
          :to="next._path"
          class="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-100 transition-colors"
        >
          {{ next.title }} →
        </NuxtLink>
      </nav>
    </ContentDoc>
  </article>
</template>

<script setup lang="ts">
const route = useRoute()

const [prev, next] = await queryContent()
  .only(['_path', 'title'])
  .findSurround(route.path)
</script>
