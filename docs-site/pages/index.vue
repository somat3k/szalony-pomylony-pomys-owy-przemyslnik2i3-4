<template>
  <!-- Landing page -->
  <div class="max-w-4xl mx-auto">
    <!-- Hero -->
    <section class="py-16 text-center">
      <div class="flex justify-center mb-6">
        <svg class="w-20 h-20 text-holo-400" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
          <polygon points="16,2 30,10 30,22 16,30 2,22 2,10" stroke="currentColor" stroke-width="2" fill="none"/>
          <polygon points="16,8 24,13 24,19 16,24 8,19 8,13" stroke="currentColor" stroke-width="1.5" fill="currentColor" fill-opacity="0.15"/>
          <circle cx="16" cy="16" r="3" fill="currentColor"/>
        </svg>
      </div>
      <h1 class="text-5xl font-bold text-gray-100 mb-4">HoloLang</h1>
      <p class="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
        A custom domain-specific language for holographic device control,
        tensor processing, and light-manipulation automation.
      </p>
      <div class="flex flex-wrap justify-center gap-4">
        <NuxtLink
          to="/getting-started/introduction"
          class="px-6 py-3 bg-holo-600 hover:bg-holo-500 text-white font-semibold rounded-lg transition-colors"
        >
          Get Started →
        </NuxtLink>
        <NuxtLink
          to="/language/syntax"
          class="px-6 py-3 bg-gray-800 hover:bg-gray-700 text-gray-100 font-semibold rounded-lg transition-colors"
        >
          Language Reference
        </NuxtLink>
      </div>
    </section>

    <!-- Feature cards -->
    <section class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pb-16">
      <FeatureCard
        v-for="feat in features"
        :key="feat.title"
        :icon="feat.icon"
        :title="feat.title"
        :desc="feat.desc"
      />
    </section>

    <!-- Quick-start code block -->
    <section class="pb-16">
      <h2 class="text-2xl font-bold text-gray-100 mb-4">Quick Start</h2>
      <div class="bg-gray-900 border border-gray-700 rounded-xl overflow-hidden">
        <div class="flex items-center gap-2 px-4 py-2 bg-gray-800 border-b border-gray-700">
          <span class="w-3 h-3 rounded-full bg-red-500" />
          <span class="w-3 h-3 rounded-full bg-yellow-500" />
          <span class="w-3 h-3 rounded-full bg-green-500" />
          <span class="ml-2 text-xs text-gray-500 font-mono">terminal</span>
        </div>
        <pre class="p-4 text-sm font-mono text-green-400 overflow-x-auto"><code>pip install hololang
hololang info          <span class="text-gray-500"># show version and subsystem overview</span>
hololang skills        <span class="text-gray-500"># list all available skills</span>
hololang run examples/hello_holo.hl
hololang repl          <span class="text-gray-500"># interactive REPL</span></code></pre>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
const features = [
  { icon: '🔆', title: 'Holographic Devices', desc: 'Control lasers, galvanized mirrors, and photonic sensors with a clean declarative syntax.' },
  { icon: '📊', title: 'Tensor Processing', desc: 'N-dimensional tensors with SafeTensor bounds-checking, computation graphs, and pooled runtimes.' },
  { icon: '🔲', title: 'Mesh MDI Canvas', desc: 'Tile-based multi-document interface with impulse cycle propagation across a sparse 2D grid.' },
  { icon: '⚙️', title: 'Kernel VM', desc: 'Stack-based kernel virtual machine with a block controller and generative pipeline.' },
  { icon: '🌐', title: 'Network Stack', desc: 'Protocol-agnostic networking: gRPC channels, WebSocket streams, webhooks, and REST APIs.' },
  { icon: '📚', title: 'Doc & Session System', desc: 'Session lifecycle management, skill registry, and hierarchical documentation directories.' },
]

useHead({ title: 'HoloLang — Holographic Device Control Language' })
</script>
