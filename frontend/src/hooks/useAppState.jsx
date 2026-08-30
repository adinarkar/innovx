/**
 * Application state shared by every page: uploaded assets, the running job and
 * its result, plus backend capability info.
 *
 * Kept in one context so the Dashboard, Processing, Match Analysis and
 * Developer pages all read the same job without refetching.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../services/api'

const AppContext = createContext(null)

const POLL_MS = 700
const SESSION_KEY = 'innovx.visualnav.session'

/**
 * The uploaded assets and the last job id are mirrored into sessionStorage so
 * a browser refresh (or opening Processing in a new tab) does not throw away a
 * completed run. Only identifiers and small metadata are stored - the result
 * itself is re-fetched from the backend.
 */
const readSession = () => {
  try {
    return JSON.parse(sessionStorage.getItem(SESSION_KEY) || 'null')
  } catch {
    return null
  }
}

const writeSession = (data) => {
  try {
    if (data) sessionStorage.setItem(SESSION_KEY, JSON.stringify(data))
    else sessionStorage.removeItem(SESSION_KEY)
  } catch {
    /* private browsing or a full quota - persistence is best-effort */
  }
}

export function AppProvider({ children }) {
  const [system, setSystem] = useState(null)
  const [mapInfo, setMapInfo] = useState(null)
  const [droneInfo, setDroneInfo] = useState(null)
  const [planInfo, setPlanInfo] = useState(null)
  const [job, setJob] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [backendUp, setBackendUp] = useState(null)
  const pollRef = useRef(null)

  // ---- backend info ----------------------------------------------------
  const refreshSystem = useCallback(async (warm = false) => {
    try {
      const info = await api.systemInfo(warm)
      setSystem(info)
      setBackendUp(true)
      return info
    } catch (err) {
      setBackendUp(false)
      return null
    }
  }, [])

  useEffect(() => {
    refreshSystem(false)
  }, [refreshSystem])

  // ---- restore the previous session on mount ---------------------------
  useEffect(() => {
    const saved = readSession()
    if (!saved) return
    if (saved.droneInfo) setDroneInfo(saved.droneInfo)
    if (saved.planInfo) setPlanInfo(saved.planInfo)

    const restore = async () => {
      if (saved.mapInfo?.map_id) {
        try {
          setMapInfo(await api.mapStatus(saved.mapInfo.map_id))
        } catch {
          setMapInfo(null)          // the backend restarted - drop the stale id
          return
        }
      }
      if (saved.jobId) {
        try {
          const status = await api.jobResult(saved.jobId)
          setJob(status)
          setResult(status.result)
        } catch {
          /* the job is gone; the uploads are still usable */
        }
      }
    }
    restore()
  }, [])

  // ---- keep the mirror in step ----------------------------------------
  useEffect(() => {
    if (!mapInfo && !droneInfo && !planInfo) {
      writeSession(null)
      return
    }
    writeSession({ mapInfo, droneInfo, planInfo, jobId: job?.job_id ?? null })
  }, [mapInfo, droneInfo, planInfo, job?.job_id])

  // ---- map indexing is asynchronous; poll until it is ready ------------
  useEffect(() => {
    if (!mapInfo?.map_id || mapInfo.embedding_status !== 'indexing') return undefined
    let cancelled = false
    const tick = async () => {
      try {
        const next = await api.mapStatus(mapInfo.map_id)
        if (!cancelled) setMapInfo(next)
        if (!cancelled && next.embedding_status === 'indexing') {
          setTimeout(tick, POLL_MS)
        }
      } catch {
        /* transient - the next user action will surface any real problem */
      }
    }
    const id = setTimeout(tick, POLL_MS)
    return () => {
      cancelled = true
      clearTimeout(id)
    }
  }, [mapInfo?.map_id, mapInfo?.embedding_status])

  // ---- uploads ---------------------------------------------------------
  const uploadMap = useCallback(async (file) => {
    setError(null)
    const info = await api.uploadMap(file)
    setMapInfo(info)
    setResult(null)
    setJob(null)
    return info
  }, [])

  const uploadDrone = useCallback(async (file) => {
    setError(null)
    const info = await api.uploadDrone(file)
    setDroneInfo(info)
    setResult(null)
    setJob(null)
    return info
  }, [])

  const uploadPlan = useCallback(async (file) => {
    setError(null)
    const info = await api.uploadPlan(file)
    setPlanInfo(info)
    return info
  }, [])

  const applyGeoreference = useCallback(
    async (body) => {
      const res = await api.setGeoreference({ ...body, map_id: mapInfo?.map_id })
      setMapInfo((prev) =>
        prev ? { ...prev, georeferenced: true, georeference: res.georeference } : prev,
      )
      return res
    },
    [mapInfo?.map_id],
  )

  const clearGeoreference = useCallback(async () => {
    if (!mapInfo?.map_id) return
    await api.clearGeoreference(mapInfo.map_id)
    setMapInfo((prev) => (prev ? { ...prev, georeferenced: false, georeference: null } : prev))
  }, [mapInfo?.map_id])

  // ---- localization ----------------------------------------------------
  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearTimeout(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const runLocalization = useCallback(
    async (options = {}) => {
      if (!mapInfo?.map_id || !droneInfo?.drone_id) {
        setError('Upload both a reference map and a drone capture first.')
        return null
      }
      setError(null)
      setResult(null)
      setBusy(true)
      stopPolling()
      try {
        const accepted = await api.localize({
          map_id: mapInfo.map_id,
          drone_id: droneInfo.drone_id,
          plan_id: planInfo?.plan_id ?? null,
          ...options,
        })
        setJob({ job_id: accepted.job_id, state: accepted.state, stages: [] })

        return await new Promise((resolve) => {
          const poll = async () => {
            try {
              const status = await api.jobStatus(accepted.job_id)
              setJob(status)
              if (status.state === 'done') {
                setResult(status.result)
                setBusy(false)
                resolve(status.result)
                return
              }
              if (status.state === 'error') {
                setError(status.error || 'Localization failed.')
                setBusy(false)
                resolve(null)
                return
              }
              pollRef.current = setTimeout(poll, POLL_MS)
            } catch (err) {
              setError(err.message)
              setBusy(false)
              resolve(null)
            }
          }
          poll()
        })
      } catch (err) {
        setError(err.message)
        setBusy(false)
        return null
      }
    },
    [mapInfo?.map_id, droneInfo?.drone_id, planInfo?.plan_id, stopPolling],
  )

  useEffect(() => stopPolling, [stopPolling])

  const reset = useCallback(() => {
    stopPolling()
    writeSession(null)
    setMapInfo(null)
    setDroneInfo(null)
    setPlanInfo(null)
    setJob(null)
    setResult(null)
    setError(null)
    setBusy(false)
  }, [stopPolling])

  const ready = Boolean(
    mapInfo?.map_id && droneInfo?.drone_id && mapInfo.embedding_status === 'ready',
  )

  const value = useMemo(
    () => ({
      system,
      backendUp,
      refreshSystem,
      mapInfo,
      droneInfo,
      planInfo,
      job,
      result,
      busy,
      error,
      ready,
      setError,
      uploadMap,
      uploadDrone,
      uploadPlan,
      applyGeoreference,
      clearGeoreference,
      runLocalization,
      reset,
    }),
    [
      system, backendUp, refreshSystem, mapInfo, droneInfo, planInfo, job, result,
      busy, error, ready, uploadMap, uploadDrone, uploadPlan, applyGeoreference,
      clearGeoreference, runLocalization, reset,
    ],
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used inside <AppProvider>')
  return ctx
}
