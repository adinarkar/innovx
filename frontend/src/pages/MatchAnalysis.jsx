import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import CandidateCard from '../components/CandidateCard'
import ImageFrame from '../components/ImageFrame'
import MapViewer from '../components/MapViewer'
import MetricCard, { Meter, MetricRow } from '../components/MetricCard'
import EmptyState from '../components/EmptyState'
import { Chip, StatusBadge } from '../components/Badge'
import { useApp } from '../hooks/useAppState'
import { apiUrl, fileUrl } from '../services/api'
import { number, percent, px, rejectionLabel, seconds } from '../utils/format'

/**
 * Side-by-side view of why the winner won: ranked candidates, the confidence
 * decomposition, and the correspondence render (spec sections 28 and 52).
 */
export default function MatchAnalysis() {
  const { result, mapInfo } = useApp()
  const navigate = useNavigate()
  const [selected, setSelected] = useState(null)
  const [inliersOnly, setInliersOnly] = useState(false)

  const candidates = result?.candidates || []
  const active = useMemo(
    () => candidates.find((c) => c.candidate_id === selected) || candidates[0],
    [candidates, selected],
  )

  if (!result) {
    return (
      <EmptyState
        title="No match analysis available"
        hint="Run a localization first — this page dissects the ranked candidates and the geometric evidence behind the chosen one."
        action={
          <button type="button" className="btn-primary mt-1" onClick={() => navigate('/')}>
            Go to Dashboard
          </button>
        }
      />
    )
  }

  const m = result.feature_metrics
  // The decision margin compares the winner against the strongest rival that
  // points somewhere *different* - an overlapping tile of the same spot is
  // corroboration, not competition (backend supplies this).
  const decision = result.decision || {}
  const anyVerified = (decision.verified_candidates ??
    candidates.filter((c) => c.homography_valid).length) > 0
  const gap = decision.margin ?? null
  const ambiguityGap = decision.ambiguity_gap ?? 0.06

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Match Analysis</h1>
          <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-ink-muted">
            {result.explanation}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={result.status} size="lg" />
          <Chip tone="neutral">{seconds(result.processing_time)}</Chip>
        </div>
      </header>

      {/* ---- decision summary ---- */}
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Winning confidence" value={percent(result.confidence, 1)} tone="brand" />
        <MetricCard label="Nearest rival"
                    value={decision.runner_up_confidence != null
                      ? percent(decision.runner_up_confidence, 1)
                      : anyVerified ? 'none' : '--'}
                    hint={decision.runner_up_tile_id != null
                      ? `Tile ${decision.runner_up_tile_id}, a different location`
                      : anyVerified
                        ? 'No competing location — the fix is unique'
                        : 'No candidate was verified'} />
        <MetricCard label="Decision margin"
                    value={!anyVerified ? '--' : gap === null ? 'unchallenged' : percent(gap, 1)}
                    tone={gap !== null && gap < ambiguityGap ? 'warn' : 'ok'}
                    hint={`Below ${percent(ambiguityGap, 0)} between two locations triggers AMBIGUOUS`} />
        <MetricCard label="Verified candidates"
                    value={`${decision.verified_candidates ?? candidates.filter((c) => c.homography_valid).length} / ${candidates.length}`}
                    hint="Passed all geometric gates" />
      </section>

      {/* ---- correspondences ---- */}
      <section className="card card-pad">
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold tracking-tight text-ink">Feature Correspondences</h2>
            <p className="mt-1 text-[12.5px] text-ink-muted">
              Drone capture on the left, best candidate tile on the right.
            </p>
          </div>
          <div className="flex overflow-hidden rounded-lg ring-1 ring-ink-line">
            {[['All Matches', false], ['RANSAC Inliers Only', true]].map(([label, value]) => (
              <button key={label} type="button" onClick={() => setInliersOnly(value)}
                      className={`px-3 py-1.5 text-[12px] font-medium transition ${
                        inliersOnly === value ? 'bg-brand text-white' : 'bg-white text-ink-muted hover:bg-brand-bg'
                      }`}>
                {label}
              </button>
            ))}
          </div>
        </div>

        {result.renders.matches_raw ? (
          <ImageFrame
            src={fileUrl(inliersOnly ? result.renders.matches_inliers : result.renders.matches_raw)}
            alt="Feature correspondences"
            aspect="aspect-[21/9]"
            caption={inliersOnly
              ? 'Only the correspondences RANSAC kept.'
              : 'Grey lines were rejected as outliers; coloured lines survived RANSAC.'}
          />
        ) : (
          <div className="grid h-56 place-items-center rounded-xl border border-dashed border-ink-line text-[13px] text-ink-muted">
            No correspondence render available for this run.
          </div>
        )}

        <div className="mt-4 grid gap-3 sm:grid-cols-4">
          <MetricCard label="Raw matches" value={number(m.raw_matches)} />
          <MetricCard label="Valid inliers" value={number(m.ransac_inliers)} tone="ok" />
          <MetricCard label="Rejected outliers"
                      value={number(Math.max(0, m.raw_matches - m.ransac_inliers))} tone="bad" />
          <MetricCard label="Reprojection error"
                      value={m.reprojection_error === null ? '--' : px(m.reprojection_error)} />
        </div>
      </section>

      {/* ---- candidate list + detail ---- */}
      <section className="grid items-start gap-4 lg:grid-cols-[1.35fr_1fr]">
        <div className="space-y-2.5">
          <h2 className="text-sm font-semibold tracking-tight text-ink">Ranked Candidates</h2>
          {candidates.map((c) => (
            <CandidateCard key={c.candidate_id} candidate={c}
                           selected={c.candidate_id === active?.candidate_id}
                           onSelect={(cand) => setSelected(cand.candidate_id)} />
          ))}
        </div>

        <div className="space-y-4">
          <h2 className="text-sm font-semibold tracking-tight text-ink">Candidate Detail</h2>
          {active ? (
            <>
              <div className="card card-pad">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-sm font-medium text-ink">
                    {active.source === 'global' ? 'Whole-map fallback' : `Tile ${active.tile_id}`}
                  </span>
                  <Chip tone={active.homography_valid ? 'ok' : 'bad'}>
                    {active.homography_valid ? 'VERIFIED' : 'REJECTED'}
                  </Chip>
                </div>
                <div className="mt-3 space-y-2.5">
                  <Meter label="Retrieval similarity" value={active.components?.retrieval} weight="0.15" />
                  <Meter label="Inlier support" value={active.components?.inliers} weight="0.30" />
                  <Meter label="Geometry quality" value={active.components?.geometry} weight="0.25" />
                  <Meter label="Spatial coverage" value={active.components?.coverage} weight="0.15" />
                  <Meter label="Ambiguity margin" value={active.components?.ambiguity} weight="0.15" />
                </div>
                <div className="mt-4 border-t border-ink-line pt-2">
                  <MetricRow label="Geometric score" value={percent(active.geometric_score, 1)} />
                  <MetricRow label="Final confidence" value={percent(active.final_score, 1)} />
                  {active.rejection && (
                    <MetricRow label="Rejection" value={rejectionLabel(active.rejection)} tone="bad" />
                  )}
                  {active.tile && (
                    <>
                      <MetricRow label="Tile origin" value={`(${active.tile.x}, ${active.tile.y})`} />
                      <MetricRow label="Tile size"
                                 value={`${active.tile.width} × ${active.tile.height} px`} />
                      <MetricRow label="Area fraction" value={percent(active.tile.scale, 0)} />
                    </>
                  )}
                </div>
              </div>

              <MapViewer
                src={apiUrl(mapInfo?.preview_url || '')}
                width={result.map_image.width}
                height={result.map_image.height}
                polygon={active.polygon}
                candidateBox={active.tile}
                candidates={[active]}
                layers={{ candidates: true, polygon: Boolean(active.polygon), center: false, keypoints: false }}
                onLayerChange={() => {}}
                viewportClass="h-[300px]"
                caption="Where this candidate sits on the reference map."
              />
            </>
          ) : (
            <div className="card card-pad text-[13px] text-ink-muted">No candidate selected.</div>
          )}
        </div>
      </section>
    </div>
  )
}
