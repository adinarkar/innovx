import { Chip, BoolBadge } from './Badge'
import { Meter } from './MetricCard'
import { percent, px, rejectionLabel } from '../utils/format'
import { fileUrl } from '../services/api'

/**
 * One row of the Candidate Analysis panel - shows enough of the decomposed
 * score that a viewer can see *why* candidate #1 won (spec section 52).
 */
export default function CandidateCard({ candidate, selected, onSelect }) {
  const c = candidate
  const preview = fileUrl(c.preview_url)
  const isBest = c.rank === 1

  return (
    <button
      type="button"
      onClick={() => onSelect?.(c)}
      className={[
        'w-full rounded-xl border bg-white p-4 text-left transition-all duration-200',
        selected
          ? 'border-brand shadow-lift ring-2 ring-brand/20'
          : 'border-ink-line shadow-card hover:border-brand-light hover:shadow-lift',
      ].join(' ')}
    >
      <div className="flex items-start gap-4">
        <div className="h-20 w-20 shrink-0 overflow-hidden rounded-lg bg-brand-bg ring-1 ring-ink-line">
          {preview ? (
            <img src={preview} alt={`Candidate ${c.rank}`} className="h-full w-full object-cover" />
          ) : (
            <div className="grid h-full place-items-center text-[10px] text-ink-muted">
              {c.source === 'global' ? 'FULL MAP' : 'n/a'}
            </div>
          )}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`font-mono text-sm font-semibold ${isBest ? 'text-brand-deep' : 'text-ink'}`}>
              #{c.rank}
            </span>
            <span className="text-sm font-medium text-ink">
              {c.source === 'global' ? 'Whole-map fallback' : `Tile ${c.tile_id}`}
            </span>
            {isBest && <Chip tone="brand">SELECTED</Chip>}
            <span className="ml-auto">
              <BoolBadge value={c.homography_valid} />
            </span>
          </div>

          <div className="mt-2 grid grid-cols-2 gap-x-5 gap-y-1 sm:grid-cols-4">
            <Stat label="Retrieval" value={percent(c.dino_similarity, 0)} />
            <Stat label="Inliers" value={`${c.inliers}/${c.raw_matches}`} />
            <Stat label="Coverage" value={percent(c.spatial_coverage, 0)} />
            <Stat label="Reproj." value={c.reprojection_error === null ? '--' : px(c.reprojection_error)} />
          </div>

          {c.rejection && (
            <p className="mt-2 text-[11.5px] text-state-bad">
              Rejected: {rejectionLabel(c.rejection)}
            </p>
          )}
          {c.rotation_applied > 0 && (
            <p className="mt-1 text-[11.5px] text-ink-muted">
              Matched after rotating the query {c.rotation_applied * 90}°.
            </p>
          )}
        </div>

        <div className="w-28 shrink-0 text-right">
          <div className="section-title">Confidence</div>
          <div className={`font-mono text-2xl font-semibold tabular-nums ${isBest ? 'text-brand-deep' : 'text-ink-muted'}`}>
            {percent(c.final_score, 1)}
          </div>
        </div>
      </div>

      {selected && c.components && (
        <div className="mt-4 grid gap-2.5 border-t border-ink-line pt-3 sm:grid-cols-2 lg:grid-cols-5">
          <Meter label="Retrieval" value={c.components.retrieval} weight="0.15" />
          <Meter label="Inliers" value={c.components.inliers} weight="0.30" />
          <Meter label="Geometry" value={c.components.geometry} weight="0.25" />
          <Meter label="Coverage" value={c.components.coverage} weight="0.15" />
          <Meter label="Ambiguity" value={c.components.ambiguity} weight="0.15" />
        </div>
      )}
    </button>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-ink-muted">{label}</div>
      <div className="font-mono text-[13px] tabular-nums text-ink-soft">{value}</div>
    </div>
  )
}
