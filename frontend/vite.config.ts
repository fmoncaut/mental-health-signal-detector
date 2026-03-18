import { defineConfig, loadEnv } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiTarget = env.VITE_API_BASE_URL || 'http://localhost:8000'

  return {
    plugins: [
      react(),
      tailwindcss(),
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    assetsInclude: ['**/*.svg', '**/*.csv'],
    server: {
      port: 5173,
      // Proxy dev uniquement — en production, configurer un reverse proxy (nginx, caddy…)
      proxy: {
        '/predict': apiTarget,
        '/checkin': apiTarget,
        '/solutions': apiTarget,
        '/analyze': apiTarget,
        '/health': apiTarget,
      },
    },
  }
})
