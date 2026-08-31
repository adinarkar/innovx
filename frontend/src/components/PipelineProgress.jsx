import { Chip } from './Badge'

/**
 * Sequential stage indicator driven by the backend's real stage log.
 *
 * The backend cannot report a meaningful percentage, so this shows genuine
 * step progression instead of a fabricated progress number (spec section 35).
 */
const PIPELINE = [
  ['prepare', 'Preparing reference map'],
  ['preprocess', 'Processing drone frame'],
  ['structure', 'Building structural representation'],
  ['translate', 'Generating map-style representation'],
  ['embed', 'Computing AI embeddings'],
  ['retrieve', 'Searching map for candidates'],
  ['features', 'Extracting local features'],
  ['match', 'Matching structural features'],
  ['verify', 'Running geometric verification'],
  ['consensus', 'Checking representation agreement'],
  ['position', 'Estimating position'],
]

export default function PipelineProgress({ job, busy }) {
  const stages = job?.stages || []
  const byKey = Object.fromEntries(stages.map((s) => [s.key, s]))
  const doneCount = stages.filter((s) => s.state === 'done' || s.state === 'skipped').length
  const pct = (doneCount / PIPELINE.length) * 100

  return (
    <section className="card card-pad animate-fade-up">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-ink">Localization Pipeline</h2>
          <p className="mt-1 text-[12px] text-ink-muted">
            Stages reported by the backend as each step completes.
          </p>
        </div>
        <Chip tone={busy ? 'brand' : doneCount === PIPELINE.length ? 'ok' : 'neutral'}>
          {busy ? 'RUNNING' : doneCount === PIPELINE.length ? 'COMPLETE' : 'IDLE'}
        </Chip>
      </div>

      <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-ink-line">
        <div
          className="h-full rounded-full bg-brand transition-[width] duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>

      <ol className="mt-5 space-y-1">
        {PIPELINE.map(([key, label], index) => {
          const stage = byKey[key]
          const state = stage?.state || 'pending'
          const isSkipped = state === 'skipped'
          const isDone = state === 'done'
          const isRunning = state === 'running'
          const isError = state === 'error'

          return (
            <li
              key={key}
              className={[
                'flex items-start gap-3 rounded-lg px-3 py-2 transition-colors duration-200',
                isRunning ? 'bg-brand-bg' : 'bg-transparent',
              ].join(' ')}
            >
              <span
                className={[
                  'mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full text-[10px] font-bold',
                  isDone
                    ? 'bg-brand text-white'
                    : isError
                      ? 'bg-state-bad text-white'
                      : isRunning
                        ? 'bg-white text-brand ring-2 ring-brand'
                        : 'bg-ink-line text-ink-muted',
                ].join(' ')}
              >
                {isDone ? '✓' : isError ? '!' : isSkipped ? '–' : index + 1}
              </span>

              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                  <span
                    className={[
                      'text-[13px]',
                      isDone || isRunning ? 'font-medium text-ink' : 'text-ink-muted',
                      isRunning ? 'animate-pulse-soft' : '',
                    ].join(' ')}
                  >
                    {label}
                    {isRunning && '...'}
                  </span>
                  {stage?.seconds > 0 && (
                    <span className="font-mono text-[11px] tabular-nums text-ink-muted">
                      {stage.seconds.toFixed(2)}s
                    </span>
                  )}
                </div>
                {stage?.detail && (
                  <p className={`mt-0.5 text-[11.5px] leading-snug ${isError ? 'text-state-bad' : 'text-ink-muted'}`}>
                    {stage.detail}
                  </p>
                )}
              </div>
            </li>
          )
        })}
      </ol>
    </section>
  )
}

export { PIPELINE }
