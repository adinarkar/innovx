import { NavLink } from 'react-router-dom'
import Brand from './Brand'
import { useApp } from '../hooks/useAppState'

const LINKS = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/processing', label: 'Processing' },
  { to: '/analysis', label: 'Match Analysis' },
  { to: '/developer', label: 'Developer' },
  { to: '/about', label: 'About' },
]

export default function Navbar() {
  const { system, backendUp } = useApp()
  const device = system?.device || '--'
  const mode = system?.app_mode || 'real'

  return (
    <header className="sticky top-0 z-40 border-b border-black/20 bg-ink/95 backdrop-blur">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-4 px-4 py-3 sm:px-6">
        <Brand />

        <nav className="order-3 -mx-1 flex w-full items-center gap-1 overflow-x-auto sm:order-2 sm:mx-0 sm:w-auto">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) =>
                [
                  'whitespace-nowrap rounded-lg px-3 py-1.5 text-[13px] font-medium transition-colors duration-150',
                  isActive
                    ? 'bg-brand text-white shadow-sm'
                    : 'text-white/65 hover:bg-white/10 hover:text-white',
                ].join(' ')
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <div className="order-2 ml-auto flex items-center gap-2 sm:order-3">
          {mode === 'demo' && (
            <span className="chip bg-state-warn/15 text-state-warn ring-1 ring-state-warn/30">
              DEMO MODE
            </span>
          )}
          <span
            className="chip bg-white/10 text-white/75"
            title="Compute device reported by the backend"
          >
            <span
              className={[
                'h-1.5 w-1.5 rounded-full',
                backendUp === false
                  ? 'bg-brand animate-pulse-soft'
                  : backendUp
                    ? 'bg-state-ok'
                    : 'bg-white/40',
              ].join(' ')}
            />
            {backendUp === false ? 'BACKEND OFFLINE' : `Device: ${device}`}
          </span>
        </div>
      </div>
    </header>
  )
}
