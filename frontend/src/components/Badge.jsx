import { STATUS_META } from '../utils/format'

const TONES = {
  ok: 'bg-state-ok/10 text-state-ok ring-1 ring-state-ok/25',
  warn: 'bg-state-warn/10 text-state-warn ring-1 ring-state-warn/25',
  bad: 'bg-brand/10 text-state-bad ring-1 ring-state-bad/25',
  neutral: 'bg-ink/5 text-ink-soft ring-1 ring-ink-line',
  brand: 'bg-brand-bg text-brand-deep ring-1 ring-brand-light',
}

export function Chip({ tone = 'neutral', children, className = '' }) {
  return <span className={`chip ${TONES[tone] || TONES.neutral} ${className}`}>{children}</span>
}

/** Coloured dot + label for one of the four localization statuses. */
export function StatusBadge({ status, size = 'md' }) {
  const meta = STATUS_META[status] || { label: status || 'Idle', tone: 'neutral' }
  const pad = size === 'lg' ? 'px-3.5 py-1.5 text-xs' : 'px-2.5 py-1 text-[11px]'
  return (
    <span className={`chip ${TONES[meta.tone]} ${pad}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {meta.label.toUpperCase()}
    </span>
  )
}

/** VALID / REJECTED style indicator used on the verification cards. */
export function BoolBadge({ value, trueLabel = 'VALID', falseLabel = 'REJECTED' }) {
  return (
    <span className={`chip ${value ? TONES.ok : TONES.bad}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {value ? trueLabel : falseLabel}
    </span>
  )
}

const VERDICT_META = {
  verified: { label: 'VERIFIED', tone: 'ok' },
  partial: { label: 'PARTIAL', tone: 'warn' },
  rejected: { label: 'REJECTED', tone: 'bad' },
}

/**
 * Three-state candidate outcome: verified / partial / rejected.
 * `partial` = every structural gate passed, only a strength threshold missed.
 * Falls back to the boolean `homography_valid` for older payloads.
 */
export function VerdictBadge({ verdict, valid }) {
  const key = verdict || (valid ? 'verified' : 'rejected')
  const meta = VERDICT_META[key] || VERDICT_META.rejected
  return (
    <span className={`chip ${TONES[meta.tone]}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {meta.label}
    </span>
  )
}
