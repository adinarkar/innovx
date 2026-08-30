import { Route, Routes } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Processing from './pages/Processing'
import MatchAnalysis from './pages/MatchAnalysis'
import Developer from './pages/Developer'
import About from './pages/About'

export default function App() {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main className="mx-auto max-w-[1400px] px-4 py-6 sm:px-6 sm:py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/processing" element={<Processing />} />
          <Route path="/analysis" element={<MatchAnalysis />} />
          <Route path="/developer" element={<Developer />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

function Footer() {
  return (
    <footer className="mt-8 border-t border-ink-line bg-brand-bg/40">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 px-4 py-5 sm:px-6">
        <p className="text-[12px] text-ink-muted">
          innov<span className="font-semibold text-brand">X</span> VisualNav — research prototype.
          Reports an estimated visual position with a stated confidence; it is not a certified
          navigation source.
        </p>
        <a href="/docs" className="text-[12px] font-medium text-brand hover:text-brand-deep"
           target="_blank" rel="noreferrer">
          API documentation →
        </a>
      </div>
    </footer>
  )
}
