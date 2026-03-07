import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import os from 'os'
import { execFileSync } from 'child_process'
import { writeFileSync, unlinkSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

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

      // Native Windows file-picker dialog — returns the full filesystem path.
      server.middlewares.use('/open-file-dialog', (_req, res) => {
        const scriptPath = join(tmpdir(), `vite-filepicker-${Date.now()}.ps1`)
        const psScript = [
          'Add-Type -AssemblyName System.Windows.Forms',
          '$d = New-Object System.Windows.Forms.OpenFileDialog',
          "$d.Title = 'Chọn file video'",
          "$d.Filter = 'Video files (*.mp4;*.avi;*.mov;*.mkv;*.wmv)|*.mp4;*.avi;*.mov;*.mkv;*.wmv|All files (*.*)|*.*'",
          'if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $d.FileName }',
        ].join('\n')
        try {
          writeFileSync(scriptPath, psScript, 'utf8')
          const result = execFileSync(
            'powershell',
            ['-NoProfile', '-STA', '-ExecutionPolicy', 'Bypass', '-File', scriptPath],
            { timeout: 60000, encoding: 'utf8' }
          ).trim()
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ path: result || null }))
        } catch {
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ path: null }))
        } finally {
          try { unlinkSync(scriptPath) } catch {}
        }
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
