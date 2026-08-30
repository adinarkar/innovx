import { Chip } from './Badge'
import { coord } from '../utils/format'

/**
 * Mission metadata from an uploaded QGroundControl .plan, plus a small
 * self-scaled plot of the waypoint path.
 *
 * The plan supplies coordinates only - the reference imagery always comes from
 * a separate upload (spec section 33).
 */
export default function MissionPanel({ plan }) {
  if (!plan) return null

  const coords = (plan.coordinates || []).filter(([a, b]) => a !== null && b !== null)
  const bounds = plan.bounds
  const path = coords.length > 1 && bounds ? buildPath(coords, bounds) : null
  const home = plan.planned_home_position

  return (
    <section className="card card-pad">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-ink">Mission Metadata</h2>
          <p className="mt-1 text-[12.5px] text-ink-muted">{plan.filename}</p>
        </div>
        <Chip tone="brand">{plan.waypoint_count} WAYPOINTS</Chip>
      </div>

      <p className="mt-3 rounded-lg bg-brand-bg px-3 py-2 text-[12px] leading-relaxed text-ink-soft ring-1 ring-brand-light">
        {plan.disclaimer}
      </p>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <dl className="space-y-1.5">
          <Row label="Ground station" value={plan.ground_station || '--'} />
          <Row label="Vehicle type" value={plan.vehicle_type ?? '--'} />
          <Row label="Cruise speed" value={plan.cruise_speed ? `${plan.cruise_speed} m/s` : '--'} />
          <Row label="Hover speed" value={plan.hover_speed ? `${plan.hover_speed} m/s` : '--'} />
          <Row
            label="Home position"
            value={home?.latitude ? `${coord(home.latitude)}, ${coord(home.longitude)}` : '--'}
          />
          <Row label="Home altitude" value={home?.altitude ? `${home.altitude} m` : '--'} />
        </dl>

        <div className="rounded-lg border border-ink-line bg-brand-bg/40 p-3">
          {path ? (
            <svg viewBox="0 0 100 100" className="h-40 w-full" preserveAspectRatio="xMidYMid meet">
              <polyline
                points={path.points}
                fill="none"
                stroke="#E57373"
                strokeWidth="1.6"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {path.nodes.map((p, i) => (
                <circle key={i} cx={p[0]} cy={p[1]} r="1.9" fill="#2B2B2B" />
              ))}
              {path.nodes[0] && (
                <circle cx={path.nodes[0][0]} cy={path.nodes[0][1]} r="3" fill="none"
                        stroke="#E57373" strokeWidth="1.2" />
              )}
            </svg>
          ) : (
            <div className="grid h-40 place-items-center text-center text-[12px] text-ink-muted">
              No plottable waypoint coordinates in this plan.
            </div>
          )}
          <p className="mt-1 text-center text-[10.5px] text-ink-muted">
            Waypoint path (relative geometry, not to scale)
          </p>
        </div>
      </div>

      {plan.warnings?.length > 0 && (
        <ul className="mt-3 space-y-1">
          {plan.warnings.map((w) => (
            <li key={w} className="text-[11.5px] text-state-warn">• {w}</li>
          ))}
        </ul>
      )}

      {coords.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-[12px] font-medium text-ink-soft hover:text-brand">
            Waypoint list
          </summary>
          <div className="mt-2 max-h-56 overflow-auto rounded-lg border border-ink-line">
            <table className="w-full text-left font-mono text-[11.5px]">
              <thead className="sticky top-0 bg-brand-bg text-ink-muted">
                <tr>
                  <th className="px-3 py-1.5 font-semibold">#</th>
                  <th className="px-3 py-1.5 font-semibold">Command</th>
                  <th className="px-3 py-1.5 font-semibold">Latitude</th>
                  <th className="px-3 py-1.5 font-semibold">Longitude</th>
                  <th className="px-3 py-1.5 font-semibold">Alt</th>
                </tr>
              </thead>
              <tbody>
                {plan.waypoints.map((w) => (
                  <tr key={w.seq} className="border-t border-ink-line">
                    <td className="px-3 py-1.5 text-ink-muted">{w.seq}</td>
                    <td className="px-3 py-1.5">{w.command_name}</td>
                    <td className="px-3 py-1.5 tabular-nums">{coord(w.latitude)}</td>
                    <td className="px-3 py-1.5 tabular-nums">{coord(w.longitude)}</td>
                    <td className="px-3 py-1.5 tabular-nums">{w.altitude ?? '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </section>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-[11px] uppercase tracking-wide text-ink-muted">{label}</dt>
      <dd className="font-mono text-[12px] text-ink-soft">{value}</dd>
    </div>
  )
}

/** Normalise lat/lon into a 0..100 SVG box, flipping latitude to screen Y. */
function buildPath(coords, bounds) {
  const spanLat = Math.max(bounds.north - bounds.south, 1e-9)
  const spanLon = Math.max(bounds.east - bounds.west, 1e-9)
  const nodes = coords.map(([lat, lon]) => [
    8 + ((lon - bounds.west) / spanLon) * 84,
    8 + ((bounds.north - lat) / spanLat) * 84,
  ])
  return { nodes, points: nodes.map((p) => p.join(',')).join(' ') }
}
