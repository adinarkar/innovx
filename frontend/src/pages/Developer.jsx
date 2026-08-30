import { useState } from 'react'
import MetricCard from '../components/MetricCard'
import { Chip, StatusBadge } from '../components/Badge'
import { useApp } from '../hooks/useAppState'
import { api } from '../services/api'
import { number, percent, seconds } from '../utils/format'

/**
 * Developer Mode: batch-score several drone frames against the loaded map and
 * report retrieval/no-match statistics (spec section 46).
 *
 * Ground truth is optional per item - a frame with no expected position still
 * runs, it just does not contribute to accuracy.
 */
export default function Developer() {
  const { mapInfo } = useApp()
  const [items, setItems] = useState([])
  const [tolerance, setTolerance] = useState(150)
  const [report, setReport] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const addFiles = async (files) => {
    setError(null)
    const picked = Array.from(files || [])
    for (const file of picked) {
      try {
        const info = await api.uploadDrone(file)
        setItems((prev) => [
          ...prev,
          {
            drone_id: info.drone_id,
            label: info.filename,
            expected_x: '',
            expected_y: '',
            expect_no_match: /unrelated|nomatch|no_match/i.test(info.filename),
          },
        ])
      } catch (err) {
        setError(err.message)
      }
    }
  }

  const update = (index, patch) =>
    setItems((prev) => prev.map((it, i) => (i === index ? { ...it, ...patch } : it)))

  const run = async () => {
    if (!mapInfo?.map_id) {
      setError('Upload a reference map on the Dashboard first.')
      return
    }
    setBusy(true)
    setError(null)
    setReport(null)
    try {
      const payload = {
        map_id: mapInfo.map_id,
        tolerance_px: Number(tolerance) || 150,
        items: items.map((it) => ({
          drone_id: it.drone_id,
          label: it.label,
          expect_no_match: it.expect_no_match,
          expected_x: it.expected_x === '' ? null : Number(it.expected_x),
          expected_y: it.expected_y === '' ? null : Number(it.expected_y),
        })),
      }
      setReport(await api.batchTest(payload))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const metrics = report?.metrics

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">Developer Mode</h1>
          <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-ink-muted">
            Score a batch of drone frames against the currently loaded reference map. Supply the
            true map pixel for a frame to count it towards accuracy, or mark it as an expected
            NO_MATCH to test false-localization behaviour.
          </p>
        </div>
        <Chip tone={mapInfo?.map_id ? 'ok' : 'bad'}>
          {mapInfo?.map_id ? `MAP ${mapInfo.map_id.slice(-6)}` : 'NO MAP LOADED'}
        </Chip>
      </header>

      <section className="card card-pad space-y-4">
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex-1">
            <span className="text-[11px] font-medium uppercase tracking-wide text-ink-muted">
              Add test frames
            </span>
            <input
              type="file"
              multiple
              accept=".png,.jpg,.jpeg,.webp"
              onChange={(e) => {
                addFiles(e.target.files)
                e.target.value = ''
              }}
              className="field mt-1 file:mr-3 file:rounded-md file:border-0 file:bg-brand file:px-3 file:py-1 file:text-[12px] file:font-semibold file:text-white hover:file:bg-brand-dark"
            />
          </label>
          <label className="w-44">
            <span className="text-[11px] font-medium uppercase tracking-wide text-ink-muted">
              Tolerance (px)
            </span>
            <input type="number" min="1" value={tolerance}
                   onChange={(e) => setTolerance(e.target.value)}
                   className="field mt-1 font-mono" />
          </label>
          <button type="button" className="btn-primary" disabled={busy || items.length === 0}
                  onClick={run}>
            {busy ? 'Running batch...' : `Run ${items.length || ''} test${items.length === 1 ? '' : 's'}`}
          </button>
          {items.length > 0 && (
            <button type="button" className="btn-ghost" onClick={() => { setItems([]); setReport(null) }}>
              Clear
            </button>
          )}
        </div>

        {error && (
          <p className="rounded-lg bg-brand-bg px-3 py-2 text-[12.5px] font-medium text-state-bad ring-1 ring-brand-light">
            {error}
          </p>
        )}

        {items.length > 0 && (
          <div className="overflow-x-auto rounded-lg border border-ink-line">
            <table className="w-full text-left text-[12.5px]">
              <thead className="bg-brand-bg text-ink-muted">
                <tr>
                  <th className="px-3 py-2 font-semibold">Frame</th>
                  <th className="px-3 py-2 font-semibold">Expected X</th>
                  <th className="px-3 py-2 font-semibold">Expected Y</th>
                  <th className="px-3 py-2 font-semibold">Expect NO_MATCH</th>
                </tr>
              </thead>
              <tbody>
                {items.map((it, i) => (
                  <tr key={it.drone_id} className="border-t border-ink-line">
                    <td className="max-w-[220px] truncate px-3 py-1.5 font-mono text-[11.5px]">
                      {it.label}
                    </td>
                    <td className="px-3 py-1.5">
                      <input type="number" value={it.expected_x} disabled={it.expect_no_match}
                             onChange={(e) => update(i, { expected_x: e.target.value })}
                             className="field !w-28 !py-1 font-mono text-[12px]" placeholder="--" />
                    </td>
                    <td className="px-3 py-1.5">
                      <input type="number" value={it.expected_y} disabled={it.expect_no_match}
                             onChange={(e) => update(i, { expected_y: e.target.value })}
                             className="field !w-28 !py-1 font-mono text-[12px]" placeholder="--" />
                    </td>
                    <td className="px-3 py-1.5">
                      <input type="checkbox" checked={it.expect_no_match}
                             onChange={(e) => update(i, { expect_no_match: e.target.checked })}
                             className="h-4 w-4 accent-[#E57373]" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {metrics && (
        <>
          <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard label="Top-1 accuracy" value={percent(metrics.top1_accuracy, 1)} tone="brand"
                        hint={`${metrics.evaluated_with_ground_truth} frames with ground truth`} />
            <MetricCard label="Top-5 retrieval" value={percent(metrics.top5_retrieval_accuracy, 1)}
                        hint="Truth inside any returned candidate" />
            <MetricCard label="No-match detection"
                        value={metrics.no_match_detection_rate === null
                          ? 'n/a' : percent(metrics.no_match_detection_rate, 1)}
                        tone="ok" hint={`${metrics.expected_no_match} expected NO_MATCH frames`} />
            <MetricCard label="False localization"
                        value={percent(metrics.false_localization_rate, 1)}
                        tone={metrics.false_localization_rate > 0 ? 'bad' : 'ok'}
                        hint="Confident but wrong" />
          </section>

          <section className="card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-[12.5px]">
                <thead className="bg-brand-bg text-ink-muted">
                  <tr>
                    {['Frame', 'Status', 'Confidence', 'Predicted', 'Expected', 'Error', 'Inliers', 'Time', 'Result']
                      .map((h) => (
                        <th key={h} className="whitespace-nowrap px-3 py-2 font-semibold">{h}</th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {report.results.map((r) => (
                    <tr key={r.drone_id} className="border-t border-ink-line hover:bg-brand-bg/40">
                      <td className="max-w-[200px] truncate px-3 py-2 font-mono text-[11.5px]">
                        {r.label}
                      </td>
                      <td className="px-3 py-2">
                        {r.status ? <StatusBadge status={r.status} /> : <Chip tone="bad">ERROR</Chip>}
                      </td>
                      <td className="px-3 py-2 font-mono tabular-nums">{percent(r.confidence, 1)}</td>
                      <td className="px-3 py-2 font-mono tabular-nums text-ink-muted">
                        {r.map_pixel ? `${r.map_pixel.x}, ${r.map_pixel.y}` : '--'}
                      </td>
                      <td className="px-3 py-2 font-mono tabular-nums text-ink-muted">
                        {r.expect_no_match ? 'NO_MATCH' : r.expected ? `${r.expected.x}, ${r.expected.y}` : '--'}
                      </td>
                      <td className="px-3 py-2 font-mono tabular-nums">
                        {r.error_px === null || r.error_px === undefined ? '--' : `${number(r.error_px, 1)} px`}
                      </td>
                      <td className="px-3 py-2 font-mono tabular-nums">{number(r.inliers)}</td>
                      <td className="px-3 py-2 font-mono tabular-nums text-ink-muted">
                        {seconds(r.processing_time)}
                      </td>
                      <td className="px-3 py-2">
                        {r.correct === null || r.correct === undefined ? (
                          <span className="text-ink-muted">--</span>
                        ) : (
                          <Chip tone={r.correct ? 'ok' : 'bad'}>{r.correct ? 'CORRECT' : 'INCORRECT'}</Chip>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      <section className="card card-pad">
        <h2 className="text-sm font-semibold tracking-tight text-ink">Suggested test set</h2>
        <p className="mt-1 text-[12.5px] leading-relaxed text-ink-muted">
          Generate one with
          <span className="mx-1 rounded bg-ink px-1.5 py-0.5 font-mono text-[11.5px] text-white">
            python test_data/generate_test_data.py
          </span>
          then upload <span className="font-mono">reference_map.jpg</span> on the Dashboard and add
          the nine test frames here. <span className="font-mono">ground_truth.json</span> lists the
          expected centre for each.
        </p>
        <ol className="mt-3 grid gap-1.5 text-[12.5px] text-ink-soft sm:grid-cols-2">
          {[
            'Direct 15% crop from the map',
            'Crop rotated 30°',
            'Crop rotated 90°',
            'Brightness and contrast shift',
            'Mild Gaussian blur',
            'Significant downscale',
            'Perspective distortion',
            'Completely unrelated image — expect NO_MATCH',
          ].map((t, i) => (
            <li key={t} className="flex gap-2">
              <span className="font-mono text-ink-muted">{i + 1}.</span>
              {t}
            </li>
          ))}
        </ol>
      </section>
    </div>
  )
}
