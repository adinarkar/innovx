import { Chip } from '../components/Badge'
import { MetricRow } from '../components/MetricCard'
import { useApp } from '../hooks/useAppState'

const PIPELINE = [
  ['Reference map', 'Multi-scale overlapping tiles (8-25% of map area, 25% overlap) embedded once and cached.'],
  ['Drone frame', 'Split into a matching branch (mild normalization only) and a visualization branch (edges, contours, structural render).'],
  ['Retrieval', 'Cosine similarity between global descriptors shortlists the top-K candidate regions. Similarity alone never decides a position.'],
  ['Local features', 'SuperPoint keypoints and descriptors, with SIFT as the fallback and debug matcher.'],
  ['Matching', 'LightGlue correspondences between the frame and each candidate tile.'],
  ['Verification', 'RANSAC homography plus convexity, scale, shear, reprojection-error and spatial-coverage gates.'],
  ['Decision', 'A weighted confidence over five components, then MATCH_FOUND / LOW_CONFIDENCE / AMBIGUOUS / NO_MATCH.'],
  ['Position', 'Frame corners and centre projected into map pixels, converted to lat/lon only if the map is georeferenced.'],
]

export default function About() {
  const { system } = useApp()
  const caps = system?.capabilities || {}

  return (
    <div className="space-y-5">
      <header className="overflow-hidden rounded-xl2 border border-ink-line bg-gradient-to-br from-brand-bg via-white to-white p-6 shadow-card sm:p-8">
        <span className="chip bg-white text-brand-deep ring-1 ring-brand-light">About</span>
        <h1 className="mt-3 text-2xl font-semibold tracking-tight text-ink sm:text-3xl">
          innov<span className="text-brand">X</span> VisualNav
        </h1>
        <p className="mt-1 text-[15px] font-medium text-ink-soft">
          AI-Powered Visual Position Recovery
        </p>
        <p className="mt-3 max-w-3xl text-[13.5px] leading-relaxed text-ink-muted">
          A drone loses GPS. A downward-facing camera still provides an aerial image. VisualNav
          analyses that scene, identifies visual and structural features, searches a stored
          reference map, verifies matching road, building and terrain geometry, rejects false
          candidate locations, and reports the most probable corresponding map area — or reports
          that it found none.
        </p>
      </header>

      <section className="card card-pad">
        <h2 className="text-sm font-semibold tracking-tight text-ink">Pipeline</h2>
        <ol className="mt-3 space-y-2.5">
          {PIPELINE.map(([title, detail], i) => (
            <li key={title} className="flex gap-3 rounded-lg border border-ink-line p-3">
              <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-brand-bg font-mono text-[11px] font-bold text-brand-deep">
                {i + 1}
              </span>
              <div>
                <div className="text-[13px] font-medium text-ink">{title}</div>
                <p className="mt-0.5 text-[12.5px] leading-relaxed text-ink-muted">{detail}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="card card-pad">
          <h2 className="text-sm font-semibold tracking-tight text-ink">Scope</h2>
          <p className="mt-2 text-[12.5px] leading-relaxed text-ink-muted">
            This prototype covers exactly one path:
          </p>
          <p className="mt-2 rounded-lg bg-ink px-3 py-2.5 font-mono text-[11.5px] leading-relaxed text-white/85">
            Drone Image → Image Processing → Candidate Map Search → Feature Matching →
            Geometric Verification → Exact Map Region → Optional GPS Coordinate
          </p>
          <p className="mt-3 text-[12.5px] leading-relaxed text-ink-muted">
            It deliberately does not implement autonomous navigation, flight-controller
            integration, SLAM, optical flow, or GPS/IMU fusion. The modular backend is arranged so
            those can be added later around an onboard computer.
          </p>
        </section>

        <section className="card card-pad">
          <h2 className="text-sm font-semibold tracking-tight text-ink">Known limitations</h2>
          <ul className="mt-2 space-y-2 text-[12.5px] leading-relaxed text-ink-muted">
            {[
              'A homography assumes a locally planar scene. Tall buildings and steep terrain introduce parallax the model cannot represent.',
              'Large appearance gaps between the reference map and the live frame — season, time of day, construction — reduce inlier counts.',
              'Repetitive layouts (identical rooftops, regular field grids) can produce genuinely ambiguous matches; the system reports AMBIGUOUS rather than guessing.',
              'The Structural Terrain View is a visualization derived from one RGB frame. It is not an elevation or 3D terrain model.',
              'Position accuracy is bounded by the reference map resolution and by the accuracy of the operator-supplied georeference.',
            ].map((line) => (
              <li key={line} className="flex gap-2">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-brand" />
                {line}
              </li>
            ))}
          </ul>
        </section>
      </div>

      <section className="card card-pad">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold tracking-tight text-ink">Runtime</h2>
          <Chip tone="brand">Device: {caps.device || '--'}</Chip>
        </div>
        <div className="mt-3 grid gap-x-8 sm:grid-cols-2">
          <div>
            <MetricRow label="Version" value={system?.version || '--'} />
            <MetricRow label="Application mode" value={system?.app_mode || '--'} />
            <MetricRow label="Retrieval backend" value={caps.retrieval_backend || '--'} />
            <MetricRow label="Matcher backend" value={caps.matcher_backend || '--'} />
          </div>
          <div>
            <MetricRow label="PyTorch" value={caps.torch_available ? 'available' : 'not installed'} />
            <MetricRow label="DINOv2" value={caps.dinov2_available ? 'loaded' : 'not loaded'} />
            <MetricRow label="SuperPoint" value={caps.superpoint_available ? 'loaded' : 'not loaded'} />
            <MetricRow label="LightGlue" value={caps.lightglue_available ? 'loaded' : 'not loaded'} />
          </div>
        </div>
        <p className="mt-4 rounded-lg bg-brand-bg px-3 py-2.5 text-[12px] leading-relaxed text-ink-soft ring-1 ring-brand-light">
          VisualNav reports an <strong>estimated visual position</strong> with a
          <strong> localization confidence</strong>. It does not guarantee a drone position and
          never returns a fabricated GPS coordinate.
        </p>
      </section>
    </div>
  )
}
