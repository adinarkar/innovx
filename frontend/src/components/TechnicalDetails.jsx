import { Chip } from './Badge'
import { MetricRow } from './MetricCard'
import { useApp } from '../hooks/useAppState'
import { number, seconds } from '../utils/format'

/**
 * Honest reporting of what is actually running: device, which models loaded,
 * and the thresholds that produced the verdict (spec sections 47 and 54).
 */
export default function TechnicalDetails({ result }) {
  const { system, refreshSystem } = useApp()
  const caps = system?.capabilities || {}
  const cfg = system?.settings || {}
  const timings = result?.timings || {}

  const models = [
    ['DINOv2', caps.dinov2_available, cfg.dino_model],
    ['SuperPoint', caps.superpoint_available, `max ${cfg.max_keypoints} keypoints`],
    ['LightGlue', caps.lightglue_available, 'SuperPoint matcher'],
    ['SIFT + RANSAC', caps.sift_available, 'OpenCV fallback / debug matcher'],
  ]

  return (
    <section className="card card-pad">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-ink">Technical Details</h2>
          <p className="mt-1 text-[12.5px] text-ink-muted">
            Active backends and thresholds for this run.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Chip tone="brand">Device: {caps.device || '--'}</Chip>
          <button type="button" className="btn-ghost !px-2.5 !py-1 text-[11px]"
                  onClick={() => refreshSystem(true)}>
            Probe models
          </button>
        </div>
      </div>

      {caps.notes?.map((note) => (
        <p key={note}
           className="mt-3 rounded-lg bg-state-warn/10 px-3 py-2 text-[12px] leading-relaxed text-state-warn">
          {note}
        </p>
      ))}

      <div className="mt-4 grid gap-5 lg:grid-cols-3">
        <div>
          <h3 className="section-title">Models</h3>
          <ul className="mt-2 space-y-1.5">
            {models.map(([name, available, detail]) => (
              <li key={name} className="flex items-center justify-between gap-3">
                <span className="flex items-center gap-2">
                  <span className={`h-1.5 w-1.5 rounded-full ${available ? 'bg-state-ok' : 'bg-ink-line'}`} />
                  <span className="text-[12.5px] text-ink">{name}</span>
                </span>
                <span className="text-right font-mono text-[11px] text-ink-muted">
                  {available ? detail : 'not loaded'}
                </span>
              </li>
            ))}
          </ul>
          <div className="mt-3 space-y-0.5">
            <MetricRow label="Retrieval backend" value={caps.retrieval_backend || '--'} />
            <MetricRow label="Matcher backend"
                       value={result?.engine?.matcher_backend || caps.matcher_backend || '--'} />
          </div>
        </div>

        <div>
          <h3 className="section-title">Thresholds</h3>
          <div className="mt-2 space-y-0.5">
            <MetricRow label="Top-K candidates" value={cfg.top_k_candidates} />
            <MetricRow label="RANSAC threshold" value={`${cfg.ransac_threshold} px`} />
            <MetricRow label="Min inliers" value={cfg.min_inliers} />
            <MetricRow label="Min inlier ratio" value={cfg.min_inlier_ratio} />
            <MetricRow label="Max reprojection error" value={`${cfg.max_reprojection_error} px`} />
            <MetricRow label="Min spatial coverage" value={cfg.min_spatial_coverage} />
            <MetricRow label="Match / low confidence"
                       value={`${cfg.match_confidence} / ${cfg.low_confidence}`} />
            <MetricRow label="Ambiguity gap" value={cfg.ambiguity_gap} />
          </div>
        </div>

        <div>
          <h3 className="section-title">This run</h3>
          <div className="mt-2 space-y-0.5">
            <MetricRow label="Map tiles searched" value={number(result?.map_image?.tiles)} />
            <MetricRow label="Working resolution" value={`${cfg.work_size} px long edge`} />
            <MetricRow label="Tile overlap" value={cfg.tile_overlap} />
            <MetricRow label="Rotation search" value={cfg.rotation_search ? 'enabled' : 'disabled'} />
            <MetricRow label="Whole-map fallback" value={cfg.global_fallback ? 'enabled' : 'disabled'} />
            {Object.entries(timings).map(([key, value]) => (
              <MetricRow key={key} label={`Stage: ${key}`} value={seconds(value)} tone="muted" />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
