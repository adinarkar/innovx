import { useState } from 'react'
import MapViewer from './MapViewer'
import MetricCard from './MetricCard'
import { Chip, StatusBadge } from './Badge'
import { fileUrl, apiUrl } from '../services/api'
import { coord, number, percent, px, rejectionLabel, STATUS_META } from '../utils/format'

/**
 * Headline result: status, confidence, the metric grid and the full map with
 * the matched region highlighted (spec sections 30 and 31).
 */
export default function ResultPanel({ result, mapInfo }) {
  const [layers, setLayers] = useState({
    polygon: true,
    center: true,
    candidates: false,
    keypoints: false,
  })

  if (!result) return null

  const meta = STATUS_META[result.status] || {}
  const m = result.feature_metrics || {}
  const accepted = result.accepted
  const best = result.best_candidate
  const tone = meta.tone === 'ok' ? 'ok' : meta.tone === 'warn' ? 'warn' : 'bad'

  return (
    <section className="space-y-5">
      {/* ---- verdict ---- */}
      <div
        className={[
          'card card-pad animate-fade-up border-l-4',
          tone === 'ok' ? 'border-l-state-ok' : tone === 'warn' ? 'border-l-state-warn' : 'border-l-state-bad',
        ].join(' ')}
      >
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-semibold tracking-tight text-ink">
                Localization Result
              </h2>
              <StatusBadge status={result.status} size="lg" />
              {result.mode === 'demo' && <Chip tone="warn">DEMO DATA</Chip>}
            </div>
            <p className="mt-1.5 max-w-2xl text-[13px] leading-relaxed text-ink-soft">
              {result.status_message}
            </p>
            <p className="mt-1 max-w-2xl text-[12.5px] leading-relaxed text-ink-muted">
              {result.explanation}
            </p>
            {result.mode_warning && (
              <p className="mt-2 rounded-lg bg-state-warn/10 px-3 py-2 text-[12px] font-medium text-state-warn">
                {result.mode_warning}
              </p>
            )}
          </div>

          <div className="text-right">
            <div className="section-title">Confidence</div>
            <div
              className={[
                'font-mono text-4xl font-semibold tabular-nums',
                tone === 'ok' ? 'text-state-ok' : tone === 'warn' ? 'text-state-warn' : 'text-state-bad',
              ].join(' ')}
            >
              {percent(result.confidence, 1)}
            </div>
            <div className="mt-0.5 text-[11px] text-ink-muted">
              {number(result.processing_time, 2)}s · {result.engine?.device}
            </div>
          </div>
        </div>
      </div>

      {/* ---- metrics ---- */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Feature Matches" value={number(m.raw_matches)}
                    hint="Correspondences before RANSAC, summed across supporting tiles" />
        <MetricCard label="RANSAC Inliers" value={number(m.ransac_inliers)}
                    tone="brand" hint="Combined evidence from every supporting tile" />
        <MetricCard label="Supporting Tiles" value={number(m.supporting_tiles)}
                    tone={m.supporting_tiles > 1 ? 'ok' : 'default'}
                    hint={m.supporting_tiles > 1
                      ? 'Independent overlapping tiles agreeing on this location'
                      : 'Only one tile verified this location'} />
        <MetricCard label="Spatial Coverage" value={percent(m.spatial_coverage)}
                    hint={`${m.coverage_cells ?? 0} of ${(m.coverage_grid ?? 4) ** 2} grid cells`} />
        <MetricCard label="Inlier Ratio" value={percent(m.inlier_ratio)}
                    hint="Inliers / raw matches" />
        <MetricCard label="Reprojection Error"
                    value={m.reprojection_error === null ? '--' : px(m.reprojection_error)}
                    hint="Inlier-weighted average across supporting tiles" />
        <MetricCard label="Homography"
                    value={m.homography_valid ? 'VALID' : 'REJECTED'}
                    tone={m.homography_valid ? 'ok' : 'bad'}
                    hint={rejectionLabel(m.rejection) || 'Passed all plausibility gates'} />
        <MetricCard label="Map Position"
                    value={result.map_pixel ? `${result.map_pixel.x}, ${result.map_pixel.y}` : '--'}
                    hint="Confidence-weighted average across supporting tiles" />
      </div>

      {/* ---- cross-representation agreement ---- */}
      {result.representation_scores?.length > 0 && (
        <div className="card card-pad">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="section-title">Cross-Representation Evidence</div>
            {result.consensus && result.consensus.participating?.length > 1 ? (
              <Chip tone={result.cross_representation_agreement ? 'ok' : 'bad'}>
                {result.cross_representation_agreement ? 'AGREEMENT: YES' : 'AGREEMENT: NO'}
              </Chip>
            ) : (
              <Chip tone="neutral">RGB ONLY</Chip>
            )}
          </div>
          <p className="mt-1 text-[12px] leading-relaxed text-ink-muted">
            Each representation is matched and geometrically verified independently.
            The RGB/geometric branch keeps sole authority over the position — an
            auxiliary branch that disagrees lowers confidence but never moves the fix.
          </p>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-[12px]">
              <thead className="text-ink-muted">
                <tr className="border-b border-ink-line">
                  <th className="py-1.5 pr-4 font-medium">Representation</th>
                  <th className="py-1.5 pr-4 font-medium">Inliers</th>
                  <th className="py-1.5 pr-4 font-medium">Inlier ratio</th>
                  <th className="py-1.5 pr-4 font-medium">Reproj. err</th>
                  <th className="py-1.5 pr-4 font-medium">Homography</th>
                  <th className="py-1.5 pr-4 font-medium">Geom. score</th>
                  <th className="py-1.5 pr-4 font-medium">Weight</th>
                  <th className="py-1.5 font-medium">Offset from RGB</th>
                </tr>
              </thead>
              <tbody className="font-mono tabular-nums text-ink">
                {result.representation_scores.map((s) => (
                  <tr key={s.representation} className="border-b border-ink-line/60">
                    <td className="py-1.5 pr-4 uppercase">{s.representation}</td>
                    <td className="py-1.5 pr-4">{number(s.inliers)}</td>
                    <td className="py-1.5 pr-4">{percent(s.inlier_ratio)}</td>
                    <td className="py-1.5 pr-4">
                      {s.reprojection_error === null ? '--' : px(s.reprojection_error)}
                    </td>
                    <td className={`py-1.5 pr-4 ${s.homography_plausible ? 'text-state-ok' : 'text-state-bad'}`}>
                      {s.homography_plausible ? 'valid' : 'rejected'}
                    </td>
                    <td className="py-1.5 pr-4">{percent(s.geometric_score)}</td>
                    <td className="py-1.5 pr-4">{percent(s.weight)}</td>
                    <td className="py-1.5">
                      {s.representation === 'rgb'
                        ? 'reference'
                        : result.consensus?.offsets_px?.[s.representation] != null
                          ? `${px(result.consensus.offsets_px[s.representation])}`
                          : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {result.consensus && result.consensus.participating?.length > 1 && (
            <p className="mt-2 font-mono text-[11px] text-ink-muted">
              tolerance {px(result.consensus.tolerance_px)} · max disagreement{' '}
              {px(result.consensus.max_disagreement_px)}
            </p>
          )}
          {result.confidence_breakdown && (
            <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 font-mono text-[11px] text-ink-muted">
              <span>RGB base: {percent(result.confidence_breakdown.base_rgb, 1)}</span>
              {result.confidence_breakdown.applied_bonus > 0 && (
                <span className="text-state-ok">
                  +{percent(result.confidence_breakdown.applied_bonus, 1)} corroboration
                  {result.confidence_breakdown.corroborating?.length
                    ? ` (${result.confidence_breakdown.corroborating.join(', ')})` : ''}
                </span>
              )}
              {result.confidence_breakdown.applied_penalty > 0 && (
                <span className="text-state-bad">
                  −{percent(result.confidence_breakdown.applied_penalty, 1)} disagreement
                  {result.confidence_breakdown.dissenting?.length
                    ? ` (${result.confidence_breakdown.dissenting.join(', ')})` : ''}
                </span>
              )}
              <span className="text-ink">
                = {percent(result.confidence_breakdown.overall, 1)} overall
              </span>
            </div>
          )}
        </div>
      )}

      {/* ---- GPS ---- */}
      <div className="card card-pad">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="section-title">GPS</div>
            {result.gps ? (
              <div className="mt-1 flex flex-wrap items-baseline gap-x-6 gap-y-1">
                <div>
                  <span className="text-[11px] text-ink-muted">Latitude </span>
                  <span className="font-mono text-lg tabular-nums text-ink">
                    {coord(result.gps.latitude)}
                  </span>
                </div>
                <div>
                  <span className="text-[11px] text-ink-muted">Longitude </span>
                  <span className="font-mono text-lg tabular-nums text-ink">
                    {coord(result.gps.longitude)}
                  </span>
                </div>
              </div>
            ) : (
              <div className="mt-1 font-mono text-lg text-ink-muted">Not Available</div>
            )}
            <p className="mt-1.5 max-w-xl text-[12px] leading-relaxed text-ink-muted">
              {result.gps
                ? 'Derived from the operator-supplied map georeference, not from imagery.'
                : 'Add a georeference for the reference map to convert the estimated map pixel into coordinates.'}
            </p>
          </div>
          <Chip tone={result.georeferenced ? 'ok' : 'neutral'}>
            {result.georeferenced ? 'GEOREFERENCED' : 'NO GEOREFERENCE'}
          </Chip>
        </div>
      </div>

      {/* ---- map + zoom ---- */}
      <div className="grid items-start gap-4 lg:grid-cols-[1.55fr_1fr]">
        <div>
          <h3 className="section-title mb-2">Full Reference Map</h3>
          <MapViewer
            src={apiUrl(mapInfo?.preview_url || '')}
            width={result.map_image?.width}
            height={result.map_image?.height}
            polygon={result.polygon}
            center={result.map_pixel}
            candidateBox={best?.tile}
            candidates={result.candidates}
            keypoints={result.inlier_map_points || []}
            layers={layers}
            onLayerChange={(key, value) => setLayers((l) => ({ ...l, [key]: value }))}
            caption={
              accepted
                ? 'Scroll to zoom, drag to pan. The pastel-red quadrilateral is the drone frame projected through the estimated homography.'
                : 'No verified region to highlight — candidate windows can still be inspected via the layer toggles.'
            }
          />
        </div>

        <div>
          <h3 className="section-title mb-2">Localized Area</h3>
          {result.renders?.localized_area ? (
            <figure className="overflow-hidden rounded-xl border border-ink-line bg-white shadow-card">
              <img
                src={fileUrl(result.renders.localized_area)}
                alt="Zoomed view of the matched map region"
                className="h-[300px] w-full bg-brand-bg/40 object-contain"
              />
              <figcaption className="border-t border-ink-line px-3.5 py-2.5 text-[12px] text-ink-muted">
                Enlarged crop around the predicted drone view.
              </figcaption>
            </figure>
          ) : (
            <div className="grid h-[300px] place-items-center rounded-xl border border-dashed border-ink-line bg-brand-bg/30 px-6 text-center text-[12.5px] text-ink-muted">
              No localized region — the pipeline did not accept a match.
            </div>
          )}

          <p className="mt-3 rounded-lg bg-brand-bg px-3 py-2.5 text-[11.5px] leading-relaxed text-ink-soft ring-1 ring-brand-light">
            {result.coverage_note}
          </p>
        </div>
      </div>
    </section>
  )
}
