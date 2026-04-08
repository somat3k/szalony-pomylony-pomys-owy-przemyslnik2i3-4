<template>
  <aside class="hidden lg:block w-64 shrink-0 sticky top-16 h-[calc(100vh-4rem)] overflow-y-auto py-8 pr-4 border-r border-gray-800">
    <nav>
      <template v-for="section in navigation" :key="section._path">
        <div class="mb-6">
          <p class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 px-3">
            {{ section.title }}
          </p>
          <ul>
            <li v-for="page in section.children" :key="page._path">
              <NuxtLink
                :to="page._path"
                class="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors"
                active-class="bg-holo-900 text-holo-300 font-medium"
                inactive-class="text-gray-400 hover:text-gray-100 hover:bg-gray-800"
              >
                {{ page.title }}
              </NuxtLink>
            </li>
          </ul>
        </div>
      </template>
    </nav>
  </aside>
</template>

<script setup lang="ts">
const { data: navigation } = await useAsyncData('navigation', () => fetchContentNavigation())
</script>
