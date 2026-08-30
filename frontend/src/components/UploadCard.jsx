import { useCallback, useRef, useState } from 'react'
import { Chip } from './Badge'
import { bytes } from '../utils/format'

/**
 * Drag-and-drop upload panel with an inline preview.
 *
 * `meta` is rendered as a definition list beneath the thumbnail so each upload
 * type can surface whatever detail matters (dimensions, aspect, waypoints).
 */
export default function UploadCard({
  label,
  description,
  accept,
  optional = false,
  file,
  preview,
  meta = [],
  status,
  onUpload,
  icon,
}) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const handleFile = useCallback(
    async (picked) => {
      if (!picked) return
      setBusy(true)
      setError(null)
      try {
        await onUpload(picked)
      } catch (err) {
        setError(err.message || 'Upload failed.')
      } finally {
        setBusy(false)
      }
    },
    [onUpload],
  )

  const onDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    handleFile(event.dataTransfer.files?.[0])
  }

  return (
    <div className="card card-pad flex h-full flex-col gap-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-brand">{icon}</span>
            <h3 className="text-sm font-semibold tracking-tight text-ink">{label}</h3>
            {optional && <Chip tone="neutral">Optional</Chip>}
          </div>
          <p className="mt-1.5 max-w-prose text-[13px] leading-relaxed text-ink-muted">
            {description}
          </p>
        </div>
        {status}
      </div>

      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={[
          'group relative flex min-h-[164px] cursor-pointer flex-col items-center justify-center gap-2',
          'rounded-xl border-2 border-dashed px-4 py-5 text-center transition-all duration-200',
          dragging
            ? 'border-brand bg-brand-bg'
            : 'border-ink-line bg-white hover:border-brand-light hover:bg-brand-bg/60',
        ].join(' ')}
      >
        {preview ? (
          <img
            src={preview}
            alt={`${label} preview`}
            className="max-h-40 w-auto rounded-lg object-contain shadow-sm ring-1 ring-ink-line"
          />
        ) : (
          <>
            <svg viewBox="0 0 24 24" className="h-8 w-8 text-ink-muted/60 transition group-hover:text-brand"
                 fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"
                 strokeLinejoin="round">
              <path d="M12 16V4m0 0L8 8m4-4l4 4" />
              <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
            </svg>
            <p className="text-[13px] font-medium text-ink-soft">
              Drop a file here or <span className="text-brand">browse</span>
            </p>
            <p className="text-[11px] text-ink-muted">{accept.replaceAll(',', '  ')}</p>
          </>
        )}

        {busy && (
          <div className="absolute inset-0 grid place-items-center rounded-xl bg-white/80">
            <div className="flex items-center gap-2 text-[13px] font-medium text-ink-soft">
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-brand border-t-transparent" />
              Uploading...
            </div>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={(e) => {
            handleFile(e.target.files?.[0])
            e.target.value = ''
          }}
        />
      </div>

      {error && (
        <p className="rounded-lg bg-brand-bg px-3 py-2 text-[12px] font-medium text-state-bad ring-1 ring-brand-light">
          {error}
        </p>
      )}

      {file && (
        <div className="space-y-2">
          <div className="flex items-baseline justify-between gap-3">
            <span className="truncate text-[13px] font-medium text-ink" title={file}>
              {file}
            </span>
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5">
            {meta.map(([key, value]) => (
              <div key={key} className="flex items-baseline justify-between gap-2">
                <dt className="text-[11px] uppercase tracking-wide text-ink-muted">{key}</dt>
                <dd className="font-mono text-[12px] tabular-nums text-ink-soft">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </div>
  )
}

export const fileSize = bytes
