import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Zoom/pan viewer for the reference map with vector overlays.
 *
 * The image keeps its natural aspect ratio and the SVG overlay uses the map's
 * pixel dimensions as its viewBox, so polygons and markers can be given
 * directly in map-pixel coordinates without any conversion in the caller.
 */
const MIN_ZOOM = 1
const MAX_ZOOM = 12

export default function MapViewer({
  src,
  width,
  height,
  polygon,
  center,
  candidateBox,
  candidates = [],
  keypoints = [],
  layers = {},
  onLayerChange,
  caption,
  viewportClass = 'h-[460px]',
  // Region selection: when regionMode is true, pointer drag draws a
  // map-pixel rectangle instead of panning. `region` renders the current
  // selection (controlled); `onRegionChange` fires with {x,y,width,height}
  // on release, or null if the drag was too small to count as a selection.
  regionMode = false,
  region = null,
  onRegionChange,
}) {
  const wrapRef = useRef(null)
  const imgRef = useRef(null)
  const [zoom, setZoom] = useState(1)
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const dragRef = useRef(null)
  const [drawRect, setDrawRect] = useState(null)
  const drawStartRef = useRef(null)

  const reset = useCallback(() => {
    setZoom(1)
    setOffset({ x: 0, y: 0 })
  }, [])

  useEffect(() => {
    reset()
  }, [src, reset])

  // Zoom towards the pointer so the feature under the cursor stays put.
  const onWheel = useCallback(
    (event) => {
      if (!wrapRef.current) return
      event.preventDefault()
      const rect = wrapRef.current.getBoundingClientRect()
      const px = event.clientX - rect.left
      const py = event.clientY - rect.top
      setZoom((prev) => {
        const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prev * (event.deltaY < 0 ? 1.15 : 1 / 1.15)))
        setOffset((o) => ({
          x: px - ((px - o.x) * next) / prev,
          y: py - ((py - o.y) * next) / prev,
        }))
        return next
      })
    },
    [],
  )

  useEffect(() => {
    const node = wrapRef.current
    if (!node) return undefined
    node.addEventListener('wheel', onWheel, { passive: false })
    return () => node.removeEventListener('wheel', onWheel)
  }, [onWheel])

  // Screen point -> natural map-pixel point, read straight off the <img>'s
  // own rendered box so it stays correct at any zoom/pan without inverting
  // the CSS transform by hand.
  const toNaturalPoint = useCallback((event) => {
    const rect = imgRef.current?.getBoundingClientRect()
    if (!rect || !width || !height) return null
    const nx = ((event.clientX - rect.left) / rect.width) * width
    const ny = ((event.clientY - rect.top) / rect.height) * height
    return { x: Math.min(Math.max(nx, 0), width), y: Math.min(Math.max(ny, 0), height) }
  }, [width, height])

  const onPointerDown = (event) => {
    if (regionMode) {
      const pt = toNaturalPoint(event)
      if (!pt) return
      drawStartRef.current = pt
      setDrawRect({ x: pt.x, y: pt.y, width: 0, height: 0 })
      event.currentTarget.setPointerCapture(event.pointerId)
      return
    }
    dragRef.current = { x: event.clientX - offset.x, y: event.clientY - offset.y }
    event.currentTarget.setPointerCapture(event.pointerId)
  }
  const onPointerMove = (event) => {
    if (regionMode) {
      if (!drawStartRef.current) return
      const pt = toNaturalPoint(event)
      if (!pt) return
      const start = drawStartRef.current
      setDrawRect({
        x: Math.min(start.x, pt.x), y: Math.min(start.y, pt.y),
        width: Math.abs(pt.x - start.x), height: Math.abs(pt.y - start.y),
      })
      return
    }
    if (!dragRef.current) return
    setOffset({ x: event.clientX - dragRef.current.x, y: event.clientY - dragRef.current.y })
  }
  const onPointerUp = (event) => {
    if (regionMode) {
      drawStartRef.current = null
      event.currentTarget.releasePointerCapture?.(event.pointerId)
      // Require a non-trivial drag (~1% of the shorter side) so a stray
      // click doesn't register as a zero-area region.
      const minSize = Math.min(width, height) * 0.01
      setDrawRect((rect) => {
        if (rect && rect.width > minSize && rect.height > minSize) {
          onRegionChange?.(rect)
        } else {
          onRegionChange?.(null)
        }
        return null
      })
      return
    }
    dragRef.current = null
    event.currentTarget.releasePointerCapture?.(event.pointerId)
  }

  const stroke = Math.max(width, height) / 320
  const toggles = [
    ['polygon', 'Predicted FOV'],
    ['center', 'Drone position'],
    ['candidates', 'Candidate regions'],
    ['keypoints', 'Matched features'],
  ]

  return (
    <div className="overflow-hidden rounded-xl border border-ink-line bg-white shadow-card">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-ink-line px-3 py-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {toggles.map(([key, label]) => (
            <button
              key={key}
              type="button"
              onClick={() => onLayerChange?.(key, !layers[key])}
              className={[
                'rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors duration-150',
                layers[key]
                  ? 'bg-brand text-white'
                  : 'bg-white text-ink-muted ring-1 ring-ink-line hover:bg-brand-bg hover:text-ink',
              ].join(' ')}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-[11px] tabular-nums text-ink-muted">
            {zoom.toFixed(1)}x
          </span>
          <button type="button" className="btn-ghost !px-2 !py-1 text-xs"
                  onClick={() => setZoom((z) => Math.min(MAX_ZOOM, z * 1.4))}>+</button>
          <button type="button" className="btn-ghost !px-2 !py-1 text-xs"
                  onClick={() => setZoom((z) => Math.max(MIN_ZOOM, z / 1.4))}>−</button>
          <button type="button" className="btn-ghost !px-2.5 !py-1 text-xs" onClick={reset}>
            Reset
          </button>
        </div>
      </div>

      <div
        ref={wrapRef}
        className={`relative ${viewportClass} overflow-hidden bg-brand-bg/50 ${
          regionMode ? 'cursor-crosshair' : 'cursor-grab active:cursor-grabbing'
        }`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <div
          className="absolute"
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom}) translate(-50%, -50%)`,
            transformOrigin: '0 0',
            left: '50%',
            top: '50%',
          }}
        >
          <div className="relative">
            <img
              ref={imgRef}
              src={src}
              alt="Reference map"
              draggable={false}
              className="max-h-[440px] w-auto select-none"
              style={{ maxWidth: '100%' }}
            />
            <svg
              viewBox={`0 0 ${width || 1} ${height || 1}`}
              className="pointer-events-none absolute inset-0 h-full w-full"
            >
              {region && (
                <rect x={region.x} y={region.y} width={region.width} height={region.height}
                      fill="#E57373" fillOpacity="0.12" stroke="#E57373"
                      strokeWidth={stroke} strokeDasharray={`${stroke * 3} ${stroke * 2}`} />
              )}
              {drawRect && (
                <rect x={drawRect.x} y={drawRect.y} width={drawRect.width} height={drawRect.height}
                      fill="#E57373" fillOpacity="0.15" stroke="#E57373" strokeWidth={stroke} />
              )}

              {layers.candidates &&
                candidates.map((c) =>
                  c.tile ? (
                    <rect
                      key={c.candidate_id}
                      x={c.tile.x}
                      y={c.tile.y}
                      width={c.tile.width}
                      height={c.tile.height}
                      fill="none"
                      stroke={c.rank === 1 ? '#E57373' : '#777777'}
                      strokeWidth={stroke * (c.rank === 1 ? 1 : 0.6)}
                      strokeDasharray={c.rank === 1 ? '' : `${stroke * 4} ${stroke * 3}`}
                      opacity={c.rank === 1 ? 0.9 : 0.55}
                    />
                  ) : null,
                )}

              {candidateBox && (
                <rect
                  x={candidateBox.x}
                  y={candidateBox.y}
                  width={candidateBox.width}
                  height={candidateBox.height}
                  fill="none"
                  stroke="#777777"
                  strokeWidth={stroke * 0.6}
                  strokeDasharray={`${stroke * 4} ${stroke * 3}`}
                />
              )}

              {layers.keypoints &&
                keypoints.map(([x, y], i) => (
                  <circle key={i} cx={x} cy={y} r={stroke * 1.1} fill="#E57373" opacity="0.75" />
                ))}

              {layers.polygon && polygon?.length >= 3 && (
                <polygon
                  points={polygon.map((p) => p.join(',')).join(' ')}
                  fill="#E57373"
                  fillOpacity="0.2"
                  stroke="#E57373"
                  strokeWidth={stroke * 1.6}
                  strokeLinejoin="round"
                />
              )}

              {layers.center && center && (
                <g>
                  <circle
                    cx={center.x}
                    cy={center.y}
                    r={stroke * 6}
                    fill="none"
                    stroke="#FFFFFF"
                    strokeWidth={stroke * 2.4}
                  />
                  <circle
                    cx={center.x}
                    cy={center.y}
                    r={stroke * 6}
                    fill="none"
                    stroke="#E57373"
                    strokeWidth={stroke * 1.6}
                  />
                  <circle cx={center.x} cy={center.y} r={stroke * 1.6} fill="#E57373" />
                  <line x1={center.x - stroke * 13} y1={center.y} x2={center.x - stroke * 7}
                        y2={center.y} stroke="#E57373" strokeWidth={stroke * 1.6} />
                  <line x1={center.x + stroke * 7} y1={center.y} x2={center.x + stroke * 13}
                        y2={center.y} stroke="#E57373" strokeWidth={stroke * 1.6} />
                  <line x1={center.x} y1={center.y - stroke * 13} x2={center.x}
                        y2={center.y - stroke * 7} stroke="#E57373" strokeWidth={stroke * 1.6} />
                  <line x1={center.x} y1={center.y + stroke * 7} x2={center.x}
                        y2={center.y + stroke * 13} stroke="#E57373" strokeWidth={stroke * 1.6} />
                </g>
              )}
            </svg>
          </div>
        </div>
      </div>

      {caption && (
        <p className="border-t border-ink-line px-3.5 py-2.5 text-[12px] text-ink-muted">
          {caption}
        </p>
      )}
    </div>
  )
}
