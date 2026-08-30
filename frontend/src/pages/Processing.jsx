import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import StageNav from '../components/StageNav'
import ImageFrame from '../components/ImageFrame'
import MapViewer from '../components/MapViewer'
import MetricCard from '../components/MetricCard'
import EmptyState from '../components/EmptyState'
import { BoolBadge, Chip, StatusBadge } from '../components/Badge'
import { useApp } from '../hooks/useAppState'
import { apiUrl, fileUrl } from '../services/api'
import { bytes, number, percent, px, rejectionLabel } from '../utils/format'

const STAGES = [
  { key: 'original', label: 'Original' },
  { key: 'corrected', label: 'Corrected' },
  { key: 'enhanced', label: 'Enhanced' },
  { key: 'structural', label: 'Structural' },
  { key: 'features', label: 'Features' },
  { key: 'candidate', label: 'Candidate' },
  { key: 'matches', label: 'Matches' },
  { key: 'verification', label: 'Verification' },
  { key: 'final', label: 'Final' },
]

export default function Processing() {
  const { result, droneInfo, mapInfo } = useApp()
  const [active, setActive] = useState('original')
  const navigate = useNavigate()

  if (!result) {
    return (
      <EmptyState
        title="No processed frame yet"
        hint="Run a localization from the Dashboard. Every stage below is rendered by the backend from your actual drone frame — nothing here is pre-baked."
        action={
          <button type="button" className="btn-primary mt-1" onClick={() => navigate('/')}>
            Go to Dashboard
          </button>
        }
      />
    )
  }

  const index = STAGES.findIndex((s) => s.key === active)

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">
            Image Processing & Structural Terrain View
          </h1>
          <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-ink-muted">
            Every transformation applied to the drone frame, in order. The matching branch keeps
            the photographic frame intact; the visualization branch produces the structural
            renders used to explain the result.
          </p>
        </div>
        <StatusBadge status={result.status} size="lg" />
      </header>

      <StageNav stages={STAGES} active={active} onSelect={setActive} />

      <div key={active} className="animate-fade-up">
        {active === 'original' && <StageOriginal result={result} droneInfo={droneInfo} />}
        {active === 'corrected' && <StageCorrected result={result} />}
        {active === 'enhanced' && <StageEnhanced result={result} />}
        {active === 'structural' && <StageStructural result={result} />}
        {active === 'features' && <StageFeatures result={result} />}
        {active === 'candidate' && <StageCandidates result={result} mapInfo={mapInfo} />}
        {active === 'matches' && <StageMatches result={result} />}
        {active === 'verification' && <StageVerification result={result} />}
        {active === 'final' && <StageFinal result={result} mapInfo={mapInfo} />}
      </div>

      <div className="flex items-center justify-between gap-3">
        <button type="button" className="btn-ghost" disabled={index === 0}
                onClick={() => setActive(STAGES[Math.max(0, index - 1)].key)}>
          ← Previous stage
        </button>
        <span className="font-mono text-[11px] text-ink-muted">
          {String(index + 1).padStart(2, '0')} / {STAGES.length}
        </span>
        <button type="button" className="btn-ghost" disabled={index === STAGES.length - 1}
                onClick={() => setActive(STAGES[Math.min(STAGES.length - 1, index + 1)].key)}>
          Next stage →
        </button>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
function StageHeader({ number: n, title, description, aside }) {
  return (
    <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h2 className="text-base font-semibold tracking-tight text-ink">
          <span className="font-mono text-brand">{n}</span> — {title}
        </h2>
        {description && (
          <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-ink-muted">{description}</p>
        )}
      </div>
      {aside}
    </div>
  )
}

function StageOriginal({ result, droneInfo }) {
  return (
    <section className="card card-pad">
      <StageHeader number="01" title="Original Drone Capture"
                   description="The uploaded frame exactly as received, before any processing." />
      <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <ImageFrame src={fileUrl(result.renders.original)} alt="Original drone capture"
                    caption="Unmodified drone frame." />
        <div className="grid grid-cols-2 gap-3 self-start">
          <MetricCard label="Filename" value={droneInfo?.filename || '--'} mono={false}
                      tone="muted" />
          <MetricCard label="File size" value={bytes(droneInfo?.file_size)} />
          <MetricCard label="Width" value={`${result.drone_image.width} px`} />
          <MetricCard label="Height" value={`${result.drone_image.height} px`} />
          <MetricCard label="Aspect ratio" value={number(result.drone_image.aspect_ratio, 3)} />
          <MetricCard label="Mean brightness"
                      value={number(result.preprocessing.mean_brightness_original, 1)} />
        </div>
      </div>
    </section>
  )
}

function StageCorrected({ result }) {
  const applied = result.preprocessing.calibration_applied
  return (
    <section className="card card-pad">
      <StageHeader
        number="02"
        title="Camera Correction"
        description={
          applied
            ? 'Lens distortion removed using the supplied intrinsics.'
            : 'No camera calibration supplied — original geometry preserved.'
        }
        aside={<Chip tone={applied ? 'ok' : 'neutral'}>{applied ? 'UNDISTORTED' : 'PASS-THROUGH'}</Chip>}
      />
      <div className="grid gap-4 sm:grid-cols-2">
        <ImageFrame src={fileUrl(result.renders.original)} alt="Before correction"
                    caption="Input frame" />
        <ImageFrame src={fileUrl(result.renders.corrected)} alt="After correction"
                    caption={applied ? 'Undistorted frame' : 'Unchanged — no intrinsics provided'} />
      </div>
      {!applied && (
        <p className="mt-3 rounded-lg bg-brand-bg px-3 py-2.5 text-[12px] leading-relaxed text-ink-soft ring-1 ring-brand-light">
          Supply <span className="font-mono">fx, fy, cx, cy, k1, k2, p1, p2, k3</span> in the
          localize request to enable undistortion. Without it the frame passes through untouched
          so no geometry is invented.
        </p>
      )}
    </section>
  )
}

function StageEnhanced({ result }) {
  const [showAfter, setShowAfter] = useState(true)
  const p = result.preprocessing
  return (
    <section className="card card-pad">
      <StageHeader
        number="03"
        title="Image Enhancement"
        description="CLAHE on the luminance channel plus brightness normalization, light denoising and a mild unsharp mask. Deliberately moderate — over-processing invents texture and inflates false matches."
        aside={
          <div className="flex overflow-hidden rounded-lg ring-1 ring-ink-line">
            {[['Before', false], ['After', true]].map(([label, value]) => (
              <button
                key={label}
                type="button"
                onClick={() => setShowAfter(value)}
                className={`px-3 py-1.5 text-[12px] font-medium transition ${
                  showAfter === value ? 'bg-brand text-white' : 'bg-white text-ink-muted hover:bg-brand-bg'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        }
      />
      <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <ImageFrame
          src={fileUrl(showAfter ? result.renders.enhanced : result.renders.corrected)}
          alt={showAfter ? 'Enhanced frame' : 'Frame before enhancement'}
          caption={showAfter ? 'CLAHE + brightness normalization + mild sharpening' : 'Input to the enhancement stage'}
        />
        <div className="grid grid-cols-2 gap-3 self-start">
          <MetricCard label="Brightness before" value={number(p.mean_brightness_original, 1)} />
          <MetricCard label="Brightness after" value={number(p.mean_brightness_enhanced, 1)} tone="brand" />
          <MetricCard label="Contrast before" value={number(p.contrast_original, 1)} />
          <MetricCard label="Contrast after" value={number(p.contrast_enhanced, 1)} tone="brand" />
          <div className="col-span-2">
            <ImageFrame src={fileUrl(result.renders.grayscale)} alt="Grayscale representation"
                        caption="Grayscale representation" aspect="aspect-[16/9]" />
          </div>
        </div>
      </div>
    </section>
  )
}

function StageStructural({ result }) {
  return (
    <section className="card card-pad">
      <StageHeader
        number="04"
        title="Structural Terrain View"
        description="AI/computer-vision derived structural representation of the aerial frame, highlighting roads, boundaries and building geometry."
        aside={<Chip tone="brand">VISUALIZATION ONLY</Chip>}
      />
      <div className="grid gap-4 lg:grid-cols-3">
        <ImageFrame src={fileUrl(result.renders.edges)} alt="Edge representation"
                    caption="Canny edge response (auto-tuned thresholds)" />
        <ImageFrame src={fileUrl(result.renders.contours)} alt="Contour overlay"
                    caption="Contours over the dimmed frame" />
        <ImageFrame src={fileUrl(result.renders.structural_map)} alt="Structural terrain view"
                    caption="Structural Terrain View — linear network in dark grey, structural contours in pastel red" />
      </div>
      <div className="mt-4 space-y-2 rounded-lg bg-brand-bg px-4 py-3 ring-1 ring-brand-light">
        <p className="text-[12.5px] leading-relaxed text-ink-soft">
          Structural representation generated from a single RGB frame. This is not a true elevation
          or 3D terrain model.
        </p>
        <p className="text-[12px] leading-relaxed text-ink-muted">
          This visualization highlights visible terrain structure such as roads, boundaries and
          building geometry. It is produced by the visualization branch and is never substituted
          for the photographic frame in the matching pipeline.
        </p>
        <p className="font-mono text-[11.5px] text-ink-muted">
          Edge density: {percent(result.preprocessing.edge_density, 2)}
        </p>
      </div>
    </section>
  )
}

function StageFeatures({ result }) {
  const k = result.keypoints
  return (
    <section className="card card-pad">
      <StageHeader
        number="05"
        title="AI Feature Extraction"
        description="Stable corners, textures, intersections and structural features used for localization."
        aside={<Chip tone="brand">{k.backend.toUpperCase()}</Chip>}
      />
      <div className="grid gap-4 lg:grid-cols-[1.5fr_1fr]">
        <ImageFrame src={fileUrl(result.renders.keypoints)} alt="Detected keypoints"
                    caption="Keypoints plotted at the working resolution used for matching." />
        <div className="grid grid-cols-2 gap-3 self-start">
          <MetricCard label="Detected keypoints" value={number(k.detected_keypoints)} />
          <MetricCard label="Selected keypoints" value={number(k.selected_keypoints)} tone="brand" />
          <MetricCard label="Working width" value={`${k.width} px`} />
          <MetricCard label="Working height" value={`${k.height} px`} />
          <div className="col-span-2 rounded-xl bg-brand-bg px-4 py-3 text-[12px] leading-relaxed text-ink-soft ring-1 ring-brand-light">
            Descriptors come from the normalized photographic frame — not from the structural
            render — so the geometry solved downstream refers to real image content.
          </div>
        </div>
      </div>
    </section>
  )
}

function StageCandidates({ result, mapInfo }) {
  const [selected, setSelected] = useState(result.candidates[0]?.candidate_id)
  const candidate = result.candidates.find((c) => c.candidate_id === selected)

  return (
    <section className="card card-pad">
      <StageHeader
        number="06"
        title="Candidate Map Search"
        description="Top candidate regions retrieved by global-descriptor similarity. Retrieval only shortlists — it never decides the final position."
        aside={<Chip tone="neutral">{result.map_image.tiles} TILES SEARCHED</Chip>}
      />

      <div className="grid gap-4 lg:grid-cols-[1fr_1.2fr]">
        <div className="space-y-2.5">
          {result.candidates.map((c) => (
            <button
              key={c.candidate_id}
              type="button"
              onClick={() => setSelected(c.candidate_id)}
              className={[
                'flex w-full items-center gap-3 rounded-xl border p-3 text-left transition-all duration-150',
                c.candidate_id === selected
                  ? 'border-brand bg-brand-bg shadow-card'
                  : 'border-ink-line bg-white hover:border-brand-light',
              ].join(' ')}
            >
              <div className="h-14 w-14 shrink-0 overflow-hidden rounded-lg bg-brand-bg ring-1 ring-ink-line">
                {c.preview_url ? (
                  <img src={fileUrl(c.preview_url)} alt={`Candidate ${c.rank}`}
                       className="h-full w-full object-cover" />
                ) : (
                  <div className="grid h-full place-items-center text-[9px] text-ink-muted">MAP</div>
                )}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-[13px] font-semibold text-brand-deep">#{c.rank}</span>
                  <span className="text-[13px] font-medium text-ink">
                    {c.source === 'global' ? 'Whole map' : `Candidate ${c.tile_id}`}
                  </span>
                </div>
                <div className="mt-0.5 text-[11.5px] text-ink-muted">
                  Similarity: {percent(c.dino_similarity, 1)}
                  {c.tile && ` · at (${c.tile.x}, ${c.tile.y}) · ${c.tile.width}px`}
                </div>
              </div>
              <BoolBadge value={c.homography_valid} />
            </button>
          ))}
        </div>

        <div>
          <MapViewer
            src={apiUrl(mapInfo?.preview_url || '')}
            width={result.map_image.width}
            height={result.map_image.height}
            candidateBox={candidate?.tile}
            candidates={candidate ? [candidate] : []}
            layers={{ candidates: true, polygon: false, center: false, keypoints: false }}
            onLayerChange={() => {}}
            viewportClass="h-[360px]"
            caption="Approximate location of the selected candidate on the full reference map."
          />
        </div>
      </div>
    </section>
  )
}

function StageMatches({ result }) {
  const [inliersOnly, setInliersOnly] = useState(false)
  const m = result.feature_metrics
  const src = inliersOnly ? result.renders.matches_inliers : result.renders.matches_raw

  return (
    <section className="card card-pad">
      <StageHeader
        number="07"
        title="Feature Matching"
        description="Correspondences between the drone frame and the best candidate tile. Grey lines were rejected by RANSAC; coloured lines are the surviving inliers."
        aside={
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
        }
      />
      {src ? (
        <ImageFrame src={fileUrl(src)} alt="Feature correspondences" aspect="aspect-[21/9]"
                    caption="Left: drone capture. Right: candidate tile." />
      ) : (
        <div className="grid h-64 place-items-center rounded-xl border border-dashed border-ink-line text-[13px] text-ink-muted">
          No correspondence render — the matcher produced no usable matches.
        </div>
      )}
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <MetricCard label="Raw matches" value={number(m.raw_matches)} />
        <MetricCard label="Valid inliers" value={number(m.ransac_inliers)} tone="ok" />
        <MetricCard label="Rejected outliers"
                    value={number(Math.max(0, m.raw_matches - m.ransac_inliers))} tone="bad" />
      </div>
    </section>
  )
}

function StageVerification({ result }) {
  const m = result.feature_metrics
  const grid = m.coverage_grid || 4
  const filled = m.coverage_cells || 0

  return (
    <section className="card card-pad">
      <StageHeader
        number="08"
        title="Geometric Verification"
        description="RANSAC estimates a homography from the correspondences, and the result is only accepted if it also passes convexity, scale, shear and coverage gates."
        aside={<BoolBadge value={m.homography_valid} />}
      />
      <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
        <div className="grid gap-3 sm:grid-cols-2">
          <MetricCard label="Feature Matches" value={number(m.raw_matches)} />
          <MetricCard label="RANSAC Inliers" value={number(m.ransac_inliers)} tone="brand" />
          <MetricCard label="Inlier Ratio" value={percent(m.inlier_ratio)} />
          <MetricCard label="Reprojection Error"
                      value={m.reprojection_error === null ? '--' : px(m.reprojection_error)} />
          <MetricCard label="Spatial Coverage" value={percent(m.spatial_coverage)} />
          <MetricCard label="Homography" value={m.homography_valid ? 'VALID' : 'REJECTED'}
                      tone={m.homography_valid ? 'ok' : 'bad'}
                      hint={rejectionLabel(m.rejection) || 'All plausibility gates passed'} />
        </div>

        <div className="rounded-xl border border-ink-line p-4">
          <h3 className="section-title">Spatial coverage grid</h3>
          <p className="mt-1 text-[12px] leading-relaxed text-ink-muted">
            The drone frame is divided into a {grid}×{grid} grid; a match supported by only a
            handful of cells is rejected as too concentrated.
          </p>
          <div className="mt-3 grid gap-1" style={{ gridTemplateColumns: `repeat(${grid}, minmax(0, 1fr))` }}>
            {Array.from({ length: grid * grid }, (_, i) => (
              <div key={i}
                   className={`aspect-square rounded ${i < filled ? 'bg-brand' : 'bg-ink-line'}`} />
            ))}
          </div>
          <p className="mt-2 font-mono text-[11.5px] text-ink-muted">
            {filled} / {grid * grid} cells contain inliers
          </p>
          {result.homography && (
            <div className="mt-4">
              <h3 className="section-title">Homography (drone px → map px)</h3>
              <pre className="mt-1.5 overflow-x-auto rounded-lg bg-ink px-3 py-2.5 font-mono text-[10.5px] leading-relaxed text-white/85">
{result.homography.map((row) => row.map((v) => v.toExponential(3).padStart(11)).join(' ')).join('\n')}
              </pre>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

function StageFinal({ result, mapInfo }) {
  const [layers, setLayers] = useState({
    polygon: true, center: true, candidates: false, keypoints: true,
  })
  return (
    <section className="card card-pad">
      <StageHeader
        number="09"
        title="Final Position"
        description="The drone frame corners projected through the homography, with the estimated centre marked on the full reference map."
        aside={<StatusBadge status={result.status} size="lg" />}
      />
      <div className="grid items-start gap-4 lg:grid-cols-[1.5fr_1fr]">
        <MapViewer
          src={apiUrl(mapInfo?.preview_url || '')}
          width={result.map_image.width}
          height={result.map_image.height}
          polygon={result.polygon}
          center={result.map_pixel}
          candidates={result.candidates}
          keypoints={result.inlier_map_points || []}
          layers={layers}
          onLayerChange={(k, v) => setLayers((l) => ({ ...l, [k]: v }))}
          caption="Full map — scroll to zoom, drag to pan."
        />
        <div className="space-y-3">
          {result.renders.localized_area ? (
            <ImageFrame src={fileUrl(result.renders.localized_area)} alt="Localized area"
                        caption="Enlarged crop of the predicted drone view." />
          ) : (
            <div className="grid h-56 place-items-center rounded-xl border border-dashed border-ink-line px-6 text-center text-[12.5px] text-ink-muted">
              No accepted region to enlarge.
            </div>
          )}
          <div className="grid grid-cols-2 gap-3">
            <MetricCard label="Map X" value={result.map_pixel ? `${result.map_pixel.x}` : '--'} />
            <MetricCard label="Map Y" value={result.map_pixel ? `${result.map_pixel.y}` : '--'} />
            <MetricCard label="Confidence" value={percent(result.confidence, 1)} tone="brand" />
            <MetricCard label="Rotation"
                        value={`${result.best_candidate?.rotation_applied_deg ?? 0}°`}
                        hint="Query rotation used" />
          </div>
        </div>
      </div>
    </section>
  )
}
