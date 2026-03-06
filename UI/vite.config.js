import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import os from 'os'

function localIPPlugin() {
  return {
    name: 'local-ip',
    configureServer(server) {
      server.middlewares.use('/local-ip', (_req, res) => {
        const interfaces = os.networkInterfaces()
        const ips = []
        for (const iface of Object.values(interfaces)) {
          for (const info of iface) {
            if (info.family === 'IPv4' && !info.internal) {
              ips.push(info.address)
            }
          }
        }
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ ips }))
      })
    }
  }
}

export default defineConfig({
  plugins: [react(), localIPPlugin()],
  server: {
    host: true,
    port: 3000
  },
  build: {
    outDir: 'dist',
    sourcemap: false
  }
})
