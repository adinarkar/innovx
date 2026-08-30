/** Textual innovX logo - engineering-oriented, with the X accented. */
export default function Brand({ compact = false }) {
  return (
    <div className="flex items-center gap-3">
      <div className="grid h-9 w-9 place-items-center rounded-lg bg-ink shadow-sm">
        <span className="font-mono text-lg font-bold leading-none text-white">
          i<span className="text-brand">X</span>
        </span>
      </div>
      <div className="leading-tight">
        <div className="text-[15px] font-semibold tracking-tight text-white">
          innov<span className="text-brand">X</span> VisualNav
        </div>
        {!compact && (
          <div className="text-[11px] font-medium tracking-wide text-white/55">
            GPS-Denied Drone Visual Localization
          </div>
        )}
      </div>
    </div>
  )
}
