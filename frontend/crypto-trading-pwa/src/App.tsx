import { useEffect } from 'react'
import { ScreenshotUpload } from './components/ScreenshotUpload'
import { NotificationPreferences } from './components/NotificationPreferences'
import { MonitoringPanel } from './components/MonitoringPanel'
import { useWebSocket } from './services/websocket'
import './App.css'

function App() {
  const { connect, disconnect, connected } = useWebSocket()

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return (
    <div className="min-h-screen bg-gray-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">
        <header className="text-center">
          <h1 className="text-3xl font-bold text-gray-900">Crypto Trading Monitor</h1>
          <p className="mt-2 text-gray-600">
            Real-time cryptocurrency trading monitoring and prediction system
          </p>
          <div className="mt-2">
            <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              connected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </header>

        <main className="grid grid-cols-1 gap-8 lg:grid-cols-12">
          <div className="lg:col-span-8">
            <section>
              <h2 className="text-xl font-semibold mb-4">Market Monitor</h2>
              <MonitoringPanel />
            </section>
          </div>

          <div className="lg:col-span-4 space-y-8">
            <section>
              <h2 className="text-xl font-semibold mb-4">Screenshot Analysis</h2>
              <ScreenshotUpload />
            </section>

            <section>
              <h2 className="text-xl font-semibold mb-4">Notification Settings</h2>
              <NotificationPreferences />
            </section>
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
