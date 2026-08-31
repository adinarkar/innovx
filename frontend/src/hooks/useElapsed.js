import { useEffect, useRef, useState } from 'react'

/**
 * Seconds elapsed since `running` last became true, updated ~5x/second.
 *
 * Freezes at its last value when `running` goes false, and resets to 0 the
 * next time it goes true. Used for the honest "this is how long it has been
 * working" readouts on the dashboard and pipeline panel - never to fake
 * progress.
 */
export function useElapsed(running) {
  const [elapsed, setElapsed] = useState(0)
  const startRef = useRef(null)

  useEffect(() => {
    if (!running) {
      startRef.current = null
      return undefined
    }
    startRef.current = performance.now()
    setElapsed(0)
    const id = setInterval(() => {
      if (startRef.current != null) {
        setElapsed((performance.now() - startRef.current) / 1000)
      }
    }, 200)
    return () => clearInterval(id)
  }, [running])

  return elapsed
}
