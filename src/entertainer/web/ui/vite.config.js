import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built bundle is served by FastAPI from web/static, which is mounted at
// /static — so asset URLs have to carry that prefix. index.html itself is
// served from "/" by a route, which is why base is absolute rather than
// relative.
export default defineConfig({
  plugins: [react()],
  base: "/static/",
  build: {
    outDir: "../static",
    // studio.html lives in web/static and is served by a separate app. An
    // emptied output directory would take it with it.
    emptyOutDir: false,
    rollupOptions: {
      output: {
        entryFileNames: "assets/[name].js",
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name].[ext]",
      },
    },
  },
  server: {
    proxy: { "/api": "http://127.0.0.1:8756" },
  },
});
