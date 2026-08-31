import { Chip } from './Badge'
import { useApp } from '../hooks/useAppState'
import { useElapsed } from '../hooks/useElapsed'

/**
 * Sequential stage indicator driven by the backend's real stage log.
 *
 * The backend cannot report a meaningful percentage, so this shows genuine
 * step progression instead of a fabricated progress number (spec section 35),
 * plus a live elapsed clock and a reassurance line when a run takes a while.
 */
const PIPELINE = [
  ['prepare', 'Preparing reference map'],
  ['preprocess', 'Processing drone frame'],
  ['embed', 'Computing region embeddings'],
  ['retrieve', 'Searching map for candidates'],
  ['features', 'Extracting local features'],
  ['match', 'Matching local features'],
  ['verify', 'Running geometric verification'],
  ['position', 'Estimating position'],
]

const RETRIEVAL_LABELS = { 'classical-embedding': 'classical descriptor', dinov2: 'DINOv2' }
const MATCHER_LABELS = { sift: 'SIFT + FLANN', 'superpoint+lightglue': 'SuperPoint + LightGlue' }

function backendSummary(caps) {
  if (!caps?.retrieval_backend) return null
  const retrieval = RETRIEVAL_LABELS[caps.retrieval_backend] || caps.retrieval_backend
  const matcher = MATCHER_LABELS[caps.matcher_backend] || caps.matcher_backend || '--'
  return `${retrieval} retrieval · ${matcher} matching`
}

export default function PipelineProgress({ job, busy }) {
  const { system } = useApp()
  const elapsed = useElapsed(busy)
  const stages = job?.stages || []
  const byKey = Object.fromEntries(stages.map((s) => [s.key, s]))
  const doneCount = stages.filter((s) => s.state === 'done').length
  const complete = doneCount === PIPELINE.length
  const pct = (doneCount / PIPELINE.length) * 100

  const shown = busy ? elapsed : (job?.elapsed_seconds ?? elapsed)
  const backend = backendSummary(system?.capabilities)

  return (
    <section className="card card-pad animate-fade-up">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-ink">Localization Pipeline</h2>
          <p className="mt-1 text-[12px] text-ink-muted">
            Stages reported by the backend as each step completes.
            {backend && <span className="ml-1 text-ink-muted/80">— {backend}</span>}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[12px] tabular-nums text-ink-muted">
            {shown.toFixed(1)}s
          </span>
          <Chip tone={busy ? 'brand' : complete ? 'ok' : 'neutral'}>
            {busy ? 'RUNNING' : complete ? 'COMPLETE' : 'IDLE'}
          </Chip>
        </div>
      </div>

      <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-ink-line">
        <div
          className="h-full rounded-full bg-brand transition-[width] duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>

      {busy && elapsed >= 8 && (
        <p className="mt-3 rounded-lg bg-brand-bg px-3 py-2 text-[12px] leading-relaxed text-ink-soft ring-1 ring-brand-light">
          {elapsed >= 25
            ? 'This frame is taking unusually long. The pipeline will still finish and return a verdict — including NO_MATCH — rather than hang.'
            : 'Larger reference maps and difficult frames take longer. The pipeline is still working.'}
        </p>
      )}

      <ol className="mt-5 space-y-1">
        {PIPELINE.map(([key, label], index) => {
          const stage = byKey[key]
          const state = stage?.state || 'pending'
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
                {isDone ? '✓' : isError ? '!' : index + 1}
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
