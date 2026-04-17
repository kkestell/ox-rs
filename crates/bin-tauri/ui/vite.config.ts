import { defineConfig } from "vite";

// Tauri hard-codes localhost:1420 in tauri.conf.json's devUrl. Make the Vite
// dev server match exactly and fail loudly if the port is taken so the
// webview doesn't end up pointing at the wrong process.
export default defineConfig({
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: "127.0.0.1",
  },
  build: {
    outDir: "dist",
    target: "es2022",
    sourcemap: true,
  },
});
