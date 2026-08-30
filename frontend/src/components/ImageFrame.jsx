import { useState } from 'react'

/**
 * Image panel with a caption, loading shimmer and graceful failure - the
 * Processing page renders a dozen of these.
 */
export default function ImageFrame({ src, alt, caption, badge, aspect = 'aspect-[4/3]', contain = true }) {
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)

  return (
    <figure className="overflow-hidden rounded-xl border border-ink-line bg-white shadow-card">
      <div className={`relative ${aspect} bg-brand-bg/40`}>
        {!loaded && !failed && (
          <div className="absolute inset-0 overflow-hidden">
            <div className="h-full w-1/3 animate-sweep bg-gradient-to-r from-transparent via-white/80 to-transparent" />
          </div>
        )}
        {failed ? (
          <div className="absolute inset-0 grid place-items-center px-4 text-center text-[12px] text-ink-muted">
            Render not available for this stage.
          </div>
        ) : (
          <img
            src={src}
            alt={alt}
            onLoad={() => setLoaded(true)}
            onError={() => setFailed(true)}
            className={`absolute inset-0 h-full w-full ${contain ? 'object-contain' : 'object-cover'} transition-opacity duration-300 ${loaded ? 'opacity-100' : 'opacity-0'}`}
          />
        )}
        {badge && <div className="absolute left-3 top-3">{badge}</div>}
      </div>
      {caption && (
        <figcaption className="border-t border-ink-line px-3.5 py-2.5 text-[12px] leading-relaxed text-ink-muted">
          {caption}
        </figcaption>
      )}
    </figure>
  )
}
