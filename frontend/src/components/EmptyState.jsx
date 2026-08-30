/** Placeholder shown when a page needs a completed job to have something to show. */
export default function EmptyState({ title, hint, action }) {
  return (
    <div className="card card-pad flex flex-col items-center gap-3 py-14 text-center">
      <div className="grid h-12 w-12 place-items-center rounded-full bg-brand-bg ring-1 ring-brand-light">
        <svg viewBox="0 0 24 24" className="h-6 w-6 text-brand" fill="none" strokeWidth="1.7"
             stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3v3m0 12v3M3 12h3m12 0h3" />
          <circle cx="12" cy="12" r="4.5" />
        </svg>
      </div>
      <h3 className="text-base font-semibold text-ink">{title}</h3>
      {hint && <p className="max-w-md text-sm leading-relaxed text-ink-muted">{hint}</p>}
      {action}
    </div>
  )
}
