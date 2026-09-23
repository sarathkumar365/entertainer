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
    // Content-hashed filenames, which is Vite's default and the reason for
    // it: FastAPI serves these as ordinary static files with no cache
    // headers, so a fixed name like assets/index.js is cached by the browser
    // across deploys and a rebuilt interface silently does not arrive.
    // index.html is served by a route and references whatever the build
    // produced, so the hash costs nothing.
  },
  server: {
    proxy: { "/api": "http://127.0.0.1:8756" },
  },
});
