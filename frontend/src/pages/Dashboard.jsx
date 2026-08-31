import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import UploadCard from '../components/UploadCard'
import PipelineProgress from '../components/PipelineProgress'
import ResultPanel from '../components/ResultPanel'
import GeoreferencePanel from '../components/GeoreferencePanel'
import MissionPanel from '../components/MissionPanel'
import TechnicalDetails from '../components/TechnicalDetails'
import CandidateCard from '../components/CandidateCard'
import { Chip } from '../components/Badge'
import { useApp } from '../hooks/useAppState'
import { useElapsed } from '../hooks/useElapsed'
import { apiUrl } from '../services/api'
import { aspect, bytes } from '../utils/format'

const IMAGE_TYPES = '.png,.jpg,.jpeg,.webp'

export default function Dashboard() {
  const {
    mapInfo, droneInfo, planInfo, job, result, busy, error, ready,
    uploadMap, uploadDrone, uploadPlan, runLocalization, reset, backendUp,
  } = useApp()
  const navigate = useNavigate()
  const [selectedCandidate, setSelectedCandidate] = useState(null)

  const indexing = mapInfo?.embedding_status === 'indexing'
  const indexElapsed = useElapsed(indexing)
  const candidates = result?.candidates || []
  const activeCandidate = selectedCandidate ?? candidates[0]?.candidate_id

  const mapMeta = useMemo(
    () =>
      mapInfo
        ? [
            ['Width', `${mapInfo.width} px`],
            ['Height', `${mapInfo.height} px`],
            ['Size', bytes(mapInfo.file_size)],
            ['Tiles', indexing ? 'building...' : mapInfo.tiles_generated],
          ]
        : [],
    [mapInfo, indexing],
  )

  const droneMeta = useMemo(
    () =>
      droneInfo
        ? [
            ['Resolution', `${droneInfo.width} × ${droneInfo.height}`],
            ['Aspect', aspect(droneInfo.aspect_ratio)],
            ['Size', bytes(droneInfo.file_size)],
            [
              'Map area',
              mapInfo
                ? `${(((droneInfo.width * droneInfo.height) / (mapInfo.width * mapInfo.height)) * 100).toFixed(1)}%`
                : '--',
            ],
          ]
        : [],
    [droneInfo, mapInfo],
  )

  return (
    <div className="space-y-6">
      <Hero onScrollToUpload={() => document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' })}
            onViewProcessing={() => navigate('/processing')} />

      {backendUp === false && (
        <div className="card card-pad border-l-4 border-l-state-bad">
          <h3 className="text-sm font-semibold text-ink">Backend unreachable</h3>
          <p className="mt-1 text-[13px] text-ink-muted">
            Start the API from the <span className="font-mono">backend/</span> directory:
            <span className="ml-1 rounded bg-ink px-1.5 py-0.5 font-mono text-[12px] text-white">
              uvicorn app.main:app --reload
            </span>
          </p>
        </div>
      )}

      {/* ---- uploads ---- */}
      <section id="upload" className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-sm font-semibold tracking-tight text-ink">Input Data</h2>
          {(mapInfo || droneInfo || planInfo) && (
            <button type="button" className="btn-ghost !py-1.5 text-[12px]" onClick={reset}>
              Clear session
            </button>
          )}
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <UploadCard
            label="Reference Map"
            description="Upload a satellite, orthomosaic, or top-view map covering the expected drone operating area."
            accept={IMAGE_TYPES}
            file={mapInfo?.filename}
            preview={mapInfo ? apiUrl(mapInfo.preview_url) : null}
            meta={mapMeta}
            onUpload={uploadMap}
            icon={<MapIcon />}
            status={
              mapInfo && (
                <Chip tone={indexing ? 'warn' : mapInfo.embedding_status === 'ready' ? 'ok' : 'bad'}>
                  {indexing ? 'INDEXING' : mapInfo.embedding_status.toUpperCase()}
                </Chip>
              )
            }
          />

          <UploadCard
            label="Drone Capture"
            description="Upload a downward-facing image captured from the drone."
            accept={IMAGE_TYPES}
            file={droneInfo?.filename}
            preview={droneInfo ? apiUrl(droneInfo.preview_url) : null}
            meta={droneMeta}
            onUpload={uploadDrone}
            icon={<DroneIcon />}
            status={droneInfo && <Chip tone="ok">READY</Chip>}
          />

          <UploadCard
            label="Mission File"
            description="Optional QGroundControl .plan. It supplies mission coordinates only — never imagery — and is not required for matching."
            accept=".plan,.json"
            optional
            file={planInfo?.filename}
            meta={
              planInfo
                ? [
                    ['Waypoints', planInfo.waypoint_count],
                    ['Home', planInfo.planned_home_position?.latitude ? 'present' : 'absent'],
                    ['Geofence', planInfo.geofence_polygons],
                    ['Rally pts', planInfo.rally_points],
                  ]
                : []
            }
            onUpload={uploadPlan}
            icon={<PlanIcon />}
            status={planInfo && <Chip tone="brand">PARSED</Chip>}
          />
        </div>

        <div className="flex flex-col items-center gap-3 pt-1">
          <button
            type="button"
            className="btn-primary w-full max-w-md !py-3 text-[15px]"
            disabled={!ready || busy}
            onClick={() => runLocalization()}
          >
            {busy ? (
              <>
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                Locating drone...
              </>
            ) : (
              <>
                <TargetIcon />
                Locate Drone
              </>
            )}
          </button>
          <p className="text-center text-[12px] text-ink-muted">
            {indexing ? (
              <>
                Building candidate regions and embeddings for the reference map…{' '}
                <span className="font-mono tabular-nums">{indexElapsed.toFixed(0)}s</span>
                {indexElapsed >= 12 && (
                  <span className="block text-ink-muted/80">
                    Large maps take longer to tile and embed — this runs once, then every
                    localization against this map is fast.
                  </span>
                )}
              </>
            ) : ready ? (
              'Runs retrieval, feature matching and geometric verification on the uploaded pair.'
            ) : (
              'Upload a reference map and a drone capture to enable localization.'
            )}
          </p>
          {error && (
            <p className="rounded-lg bg-brand-bg px-4 py-2 text-[12.5px] font-medium text-state-bad ring-1 ring-brand-light">
              {error}
            </p>
          )}
        </div>
      </section>

      {(busy || job) && <PipelineProgress job={job} busy={busy} />}

      {result && (
        <>
          <ResultPanel result={result} mapInfo={mapInfo} />

          <section className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold tracking-tight text-ink">Candidate Analysis</h2>
                <p className="mt-1 text-[12.5px] text-ink-muted">
                  Every shortlisted region with its decomposed score — click a card to expand the breakdown.
                </p>
              </div>
              <button type="button" className="btn-ghost !py-1.5 text-[12px]"
                      onClick={() => navigate('/analysis')}>
                Open Match Analysis
              </button>
            </div>
            <div className="space-y-2.5">
              {candidates.map((c) => (
                <CandidateCard
                  key={c.candidate_id}
                  candidate={c}
                  selected={c.candidate_id === activeCandidate}
                  onSelect={(cand) => setSelectedCandidate(cand.candidate_id)}
                />
              ))}
            </div>
          </section>
        </>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <GeoreferencePanel />
        {planInfo ? (
          <MissionPanel plan={planInfo} />
        ) : (
          <section className="card card-pad">
            <h2 className="text-sm font-semibold tracking-tight text-ink">Mission Metadata</h2>
            <p className="mt-2 text-[12.5px] leading-relaxed text-ink-muted">
              Upload a QGroundControl <span className="font-mono">.plan</span> to display the planned
              home position, waypoints and altitudes alongside the visual fix. The plan contains
              mission coordinates only — it does not contain satellite imagery, so the reference map
              still comes from its own upload.
            </p>
          </section>
        )}
      </div>

      <TechnicalDetails result={result} />
    </div>
  )
}

/* ------------------------------------------------------------------ */
function Hero({ onScrollToUpload, onViewProcessing }) {
  return (
    <section className="overflow-hidden rounded-xl2 border border-ink-line bg-gradient-to-br from-brand-bg via-white to-white shadow-card">
      <div className="grid gap-6 p-6 sm:p-8 lg:grid-cols-[1.5fr_1fr] lg:items-center">
        <div>
          <span className="chip bg-white text-brand-deep ring-1 ring-brand-light">
            innovX · Visual Localization Prototype
          </span>
          <h1 className="mt-3 text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
            innov<span className="text-brand">X</span> VisualNav
          </h1>
          <p className="mt-1.5 text-base font-medium text-ink-soft">
            GPS-Denied Drone Visual Localization
          </p>
          <p className="mt-3 max-w-2xl text-[14px] leading-relaxed text-ink-muted">
            Recovering drone position by matching live aerial imagery against stored reference
            maps. The pipeline searches candidate map regions, matches structural features,
            verifies the geometry with RANSAC, and reports an estimated position with a
            confidence it can also refuse to give.
          </p>
          <div className="mt-5 flex flex-wrap gap-2.5">
            <button type="button" className="btn-primary" onClick={onScrollToUpload}>
              Upload Data
            </button>
            <button type="button" className="btn-ghost" onClick={onViewProcessing}>
              View Processing
            </button>
          </div>
        </div>

        <ul className="grid gap-2 text-[12.5px] text-ink-soft">
          {[
            'Multi-scale overlapping map tiling with cached embeddings',
            'Global descriptor retrieval shortlists candidate regions',
            'Local feature matching + RANSAC homography verification',
            'Explicit MATCH / LOW CONFIDENCE / AMBIGUOUS / NO MATCH verdicts',
            'GPS only when the map carries an operator-supplied georeference',
          ].map((line) => (
            <li key={line} className="flex items-start gap-2.5 rounded-lg bg-white/70 px-3 py-2 ring-1 ring-ink-line">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-brand" />
              {line}
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

/* --- inline icons keep the bundle free of an icon dependency --- */
const iconProps = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.7,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  className: 'h-4 w-4',
}

const MapIcon = () => (
  <svg {...iconProps}>
    <path d="M9 4L3 6v14l6-2 6 2 6-2V4l-6 2-6-2z" />
    <path d="M9 4v14M15 6v14" />
  </svg>
)
const DroneIcon = () => (
  <svg {...iconProps}>
    <rect x="9" y="9" width="6" height="6" rx="1.5" />
    <path d="M9 9L5 5M15 9l4-4M9 15l-4 4M15 15l4 4" />
    <circle cx="4" cy="4" r="2" />
    <circle cx="20" cy="4" r="2" />
    <circle cx="4" cy="20" r="2" />
    <circle cx="20" cy="20" r="2" />
  </svg>
)
const PlanIcon = () => (
  <svg {...iconProps}>
    <path d="M5 4h9l5 5v11a1 1 0 01-1 1H5a1 1 0 01-1-1V5a1 1 0 011-1z" />
    <path d="M14 4v5h5M8 13h8M8 17h5" />
  </svg>
)
const TargetIcon = () => (
  <svg {...iconProps} className="h-4 w-4">
    <circle cx="12" cy="12" r="7" />
    <circle cx="12" cy="12" r="2.5" />
    <path d="M12 2v3M12 19v3M2 12h3M19 12h3" />
  </svg>
)
