/** Single metric tile used across the result and verification panels. */
export default function MetricCard({ label, value, hint, tone = 'default', mono = true }) {
  const toneClass = {
    default: 'text-ink',
    brand: 'text-brand-deep',
    ok: 'text-state-ok',
    warn: 'text-state-warn',
    bad: 'text-state-bad',
    muted: 'text-ink-muted',
  }[tone]

  return (
    <div className="rounded-xl border border-ink-line bg-white px-4 py-3.5 transition hover:border-brand-light hover:shadow-card">
      <div className="section-title">{label}</div>
      <div className={`mt-1.5 metric-value ${toneClass} ${mono ? 'font-mono' : ''}`}>{value}</div>
      {hint && <div className="mt-1 text-[11px] leading-snug text-ink-muted">{hint}</div>}
    </div>
  )
}

/** Compact label/value row for dense diagnostic lists. */
export function MetricRow({ label, value, tone = 'default' }) {
  const toneClass = {
    default: 'text-ink',
    ok: 'text-state-ok',
    warn: 'text-state-warn',
    bad: 'text-state-bad',
    muted: 'text-ink-muted',
  }[tone]
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="text-[12px] text-ink-muted">{label}</span>
      <span className={`font-mono text-[12px] tabular-nums ${toneClass}`}>{value}</span>
    </div>
  )
}

/** Horizontal 0..1 meter, used for confidence component breakdowns. */
export function Meter({ label, value, weight }) {
  const pct = Math.max(0, Math.min(1, Number(value) || 0)) * 100
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[12px] text-ink-soft">
          {label}
          {weight !== undefined && (
            <span className="ml-1.5 text-[10px] text-ink-muted">w={weight}</span>
          )}
        </span>
        <span className="font-mono text-[12px] tabular-nums text-ink">{pct.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-ink-line">
        <div
          className="h-full rounded-full bg-brand transition-[width] duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
