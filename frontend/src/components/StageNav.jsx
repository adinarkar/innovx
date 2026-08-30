/** Horizontal, clickable step navigation for the Processing page. */
export default function StageNav({ stages, active, onSelect, disabled = false }) {
  return (
    <div className="-mx-1 flex items-center gap-1 overflow-x-auto pb-1">
      {stages.map((stage, index) => {
        const isActive = stage.key === active
        return (
          <div key={stage.key} className="flex shrink-0 items-center">
            <button
              type="button"
              disabled={disabled}
              onClick={() => onSelect(stage.key)}
              className={[
                'flex items-center gap-2 rounded-lg px-3 py-2 text-[12.5px] font-medium transition-all duration-150',
                'disabled:cursor-not-allowed disabled:opacity-45',
                isActive
                  ? 'bg-brand text-white shadow-sm'
                  : 'bg-white text-ink-muted ring-1 ring-ink-line hover:bg-brand-bg hover:text-ink',
              ].join(' ')}
            >
              <span className={`font-mono text-[10px] ${isActive ? 'text-white/70' : 'text-ink-muted'}`}>
                {String(index + 1).padStart(2, '0')}
              </span>
              {stage.label}
            </button>
            {index < stages.length - 1 && (
              <svg viewBox="0 0 24 24" className="mx-0.5 h-3.5 w-3.5 shrink-0 text-ink-line"
                   fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M9 6l6 6-6 6" />
              </svg>
            )}
          </div>
        )
      })}
    </div>
  )
}
