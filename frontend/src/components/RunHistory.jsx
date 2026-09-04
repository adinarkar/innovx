import { useState } from 'react'
import MetricCard from './MetricCard'
import { Chip, StatusBadge } from './Badge'
import { useApp } from '../hooks/useAppState'
import { percent, seconds } from '../utils/format'

/**
 * Client-side run history: every completed localization this session, with a
 * checkbox-driven comparison strip so computing time, confidence and the
 * winning candidate can be compared across runs (e.g. Efficient Matching
 * on vs off, or different search regions). Nothing here is re-fetched from
 * the backend - every field already comes back on the job result.
 */
export default function RunHistory() {
  const { history, clearHistory } = useApp()
  const [selected, setSelected] = useState(() => new Set())

  const toggle = (id) =>
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  if (history.length === 0) {
    return (
      <section className="card card-pad">
        <h2 className="text-sm font-semibold tracking-tight text-ink">Run History</h2>
        <p className="mt-2 text-[12.5px] leading-relaxed text-ink-muted">
          Every localization you run this session is kept here so you can select two or more and
          compare their computing time, confidence and winning candidate side by side.
        </p>
      </section>
    )
  }

  const compared = history.filter((h) => selected.has(h.id))

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-ink">Run History</h2>
          <p className="mt-1 text-[12.5px] text-ink-muted">
            Check two or more runs to compare them.
          </p>
        </div>
        <button type="button" className="btn-ghost !py-1.5 text-[12px]" onClick={clearHistory}>
          Clear history
        </button>
      </div>

      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-[12.5px]">
            <thead className="bg-brand-bg text-ink-muted">
              <tr>
                <th className="px-3 py-2 font-semibold" />
                <th className="px-3 py-2 font-semibold">Run</th>
                <th className="px-3 py-2 font-semibold">Options</th>
                <th className="px-3 py-2 font-semibold">Status</th>
                <th className="px-3 py-2 font-semibold">Confidence</th>
                <th className="px-3 py-2 font-semibold">Time</th>
                <th className="px-3 py-2 font-semibold">Candidate</th>
              </tr>
            </thead>
            <tbody>
              {history.map((h) => (
                <tr key={h.id} className="border-t border-ink-line hover:bg-brand-bg/40">
                  <td className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={selected.has(h.id)}
                      onChange={() => toggle(h.id)}
                      className="h-4 w-4 accent-[#E57373]"
                    />
                  </td>
                  <td className="px-3 py-2 font-mono text-[11.5px] text-ink-muted">
                    <div>{h.id.slice(-8)}</div>
                    <div>{new Date(h.ranAt).toLocaleTimeString()}</div>
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex flex-wrap gap-1">
                      {h.options?.efficient_features && <Chip tone="brand">Efficient</Chip>}
                      {h.options?.search_region && <Chip tone="neutral">Region-limited</Chip>}
                      {h.options?.matcher && <Chip tone="neutral">{h.options.matcher}</Chip>}
                      {!h.options?.efficient_features && !h.options?.search_region &&
                        !h.options?.matcher && <span className="text-ink-muted">default</span>}
                    </div>
                  </td>
                  <td className="px-3 py-2"><StatusBadge status={h.status} /></td>
                  <td className="px-3 py-2 font-mono tabular-nums">{percent(h.confidence, 1)}</td>
                  <td className="px-3 py-2 font-mono tabular-nums text-ink-muted">
                    {seconds(h.processingTime)}
                  </td>
                  <td className="px-3 py-2 font-mono tabular-nums text-ink-muted">
                    {h.bestCandidate?.tile_id ?? '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {compared.length >= 2 && (
        <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${compared.length}, minmax(0, 1fr))` }}>
          {compared.map((h) => (
            <div key={h.id} className="space-y-2">
              <div className="flex items-center justify-between gap-2 px-1">
                <span className="font-mono text-[11px] text-ink-muted">{h.id.slice(-8)}</span>
                <StatusBadge status={h.status} />
              </div>
              <MetricCard label="Confidence" value={percent(h.confidence, 1)} tone="brand" />
              <MetricCard label="Computing time" value={seconds(h.processingTime)} />
              <MetricCard label="Selected candidate"
                          value={h.bestCandidate?.tile_id !== undefined ? `Tile ${h.bestCandidate.tile_id}` : '--'}
                          hint={h.options?.efficient_features ? 'Efficient matching' : 'Default matching'} />
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
