import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // During dev, forward /api/* to the Express backend on port 3001.
      // In production you'd deploy both behind the same domain or a reverse proxy.
      '/api': 'http://localhost:3001',
    },
  },
})
