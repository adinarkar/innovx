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
                    hint="Correspondences before RANSAC" />
        <MetricCard label="RANSAC Inliers" value={number(m.ransac_inliers)}
                    tone="brand" hint="Geometrically consistent matches" />
        <MetricCard label="Inlier Ratio" value={percent(m.inlier_ratio)}
                    hint="Inliers / raw matches" />
        <MetricCard label="Spatial Coverage" value={percent(m.spatial_coverage)}
                    hint={`${m.coverage_cells ?? 0} of ${(m.coverage_grid ?? 4) ** 2} grid cells`} />
        <MetricCard label="Reprojection Error"
                    value={m.reprojection_error === null ? '--' : px(m.reprojection_error)}
                    hint="Mean over inliers" />
        <MetricCard label="Homography"
                    value={m.homography_valid ? 'VALID' : 'REJECTED'}
                    tone={m.homography_valid ? 'ok' : 'bad'}
                    hint={rejectionLabel(m.rejection) || 'Passed all plausibility gates'} />
        <MetricCard label="Map X"
                    value={result.map_pixel ? `${result.map_pixel.x} px` : '--'}
                    hint="Estimated drone centre" />
        <MetricCard label="Map Y"
                    value={result.map_pixel ? `${result.map_pixel.y} px` : '--'}
                    hint="Estimated drone centre" />
      </div>

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
