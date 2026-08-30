import { useEffect, useState } from 'react'
import { Chip } from './Badge'
import { useApp } from '../hooks/useAppState'
import { coord } from '../utils/format'

/**
 * Optional lat/lon extent for the reference map.
 *
 * Without this the system reports map pixels only - it never derives GPS from
 * imagery (spec sections 32 and 45).
 */
export default function GeoreferencePanel() {
  const { mapInfo, planInfo, applyGeoreference, clearGeoreference } = useApp()
  const [values, setValues] = useState({ north: '', south: '', west: '', east: '' })
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const geo = mapInfo?.georeference

  useEffect(() => {
    if (geo?.kind === 'bbox') {
      setValues({
        north: String(geo.north), south: String(geo.south),
        west: String(geo.west), east: String(geo.east),
      })
    }
  }, [geo?.kind, geo?.north, geo?.south, geo?.west, geo?.east])

  const suggestion = planInfo?.suggested_georeference

  const submit = async (event) => {
    event.preventDefault()
    setError(null)
    setBusy(true)
    try {
      await applyGeoreference({
        north: Number(values.north), south: Number(values.south),
        west: Number(values.west), east: Number(values.east),
      })
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const fields = [
    ['north', 'North latitude', '12.9760'],
    ['south', 'South latitude', '12.9580'],
    ['west', 'West longitude', '77.5880'],
    ['east', 'East longitude', '77.6080'],
  ]

  return (
    <section className="card card-pad">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight text-ink">
            Map Georeference <span className="font-normal text-ink-muted">(optional)</span>
          </h2>
          <p className="mt-1 max-w-prose text-[12.5px] leading-relaxed text-ink-muted">
            Supply the geographic extent of the reference map to convert the estimated
            map pixel into latitude/longitude. Coordinates are never inferred from imagery.
          </p>
        </div>
        <Chip tone={mapInfo?.georeferenced ? 'ok' : 'neutral'}>
          {mapInfo?.georeferenced ? 'GEOREFERENCED' : 'PIXELS ONLY'}
        </Chip>
      </div>

      {suggestion && (
        <button
          type="button"
          onClick={() =>
            setValues({
              north: String(suggestion.north), south: String(suggestion.south),
              west: String(suggestion.west), east: String(suggestion.east),
            })
          }
          className="mt-3 w-full rounded-lg bg-brand-bg px-3 py-2 text-left text-[12px] text-ink-soft ring-1 ring-brand-light transition hover:bg-brand-light/40"
        >
          <span className="font-semibold text-brand-deep">Use mission extent</span> from the
          uploaded .plan — N {coord(suggestion.north, 4)}, S {coord(suggestion.south, 4)},
          W {coord(suggestion.west, 4)}, E {coord(suggestion.east, 4)}.
          <span className="block text-[11px] text-ink-muted">
            A suggestion only: confirm it matches the area your map image actually covers.
          </span>
        </button>
      )}

      <form onSubmit={submit} className="mt-4 space-y-3">
        <div className="grid grid-cols-2 gap-3">
          {fields.map(([key, label, placeholder]) => (
            <label key={key} className="block">
              <span className="text-[11px] font-medium uppercase tracking-wide text-ink-muted">
                {label}
              </span>
              <input
                className="field mt-1 font-mono"
                type="number"
                step="any"
                required
                placeholder={placeholder}
                value={values[key]}
                onChange={(e) => setValues((v) => ({ ...v, [key]: e.target.value }))}
              />
            </label>
          ))}
        </div>

        {error && (
          <p className="rounded-lg bg-brand-bg px-3 py-2 text-[12px] font-medium text-state-bad ring-1 ring-brand-light">
            {error}
          </p>
        )}

        <div className="flex flex-wrap items-center gap-2">
          <button type="submit" className="btn-primary" disabled={!mapInfo?.map_id || busy}>
            {busy ? 'Applying...' : 'Apply georeference'}
          </button>
          {mapInfo?.georeferenced && (
            <button type="button" className="btn-ghost" onClick={clearGeoreference}>
              Clear
            </button>
          )}
          {geo?.ground_sample_distance_m && (
            <span className="text-[11.5px] text-ink-muted">
              ≈ {geo.ground_sample_distance_m} m / pixel
            </span>
          )}
        </div>
      </form>
    </section>
  )
}
