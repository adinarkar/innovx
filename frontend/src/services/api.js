/**
 * Thin client for the innovX VisualNav FastAPI backend.
 *
 * In development the Vite proxy forwards /api and /files to the backend, so
 * BASE stays empty. Set VITE_API_BASE when the two are deployed separately.
 */
const BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '')

export const apiUrl = (path) => `${BASE}${path}`

/** Absolute URL for a render written under processed/{job_id}/. */
export const fileUrl = (relative) =>
  relative ? apiUrl(`/files/processed/${relative}`) : null

class ApiError extends Error {
  constructor(message, status, payload) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

async function parse(response) {
  const text = await response.text()
  let payload = null
  try {
    payload = text ? JSON.parse(text) : null
  } catch {
    payload = { detail: text }
  }
  if (!response.ok) {
    const message =
      payload?.detail || payload?.error || `Request failed (${response.status})`
    throw new ApiError(
      typeof message === 'string' ? message : JSON.stringify(message),
      response.status,
      payload,
    )
  }
  return payload
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(apiUrl(path), options)
  } catch (err) {
    throw new ApiError(
      'Cannot reach the backend. Start it with: uvicorn app.main:app --reload',
      0,
      null,
    )
  }
  return parse(response)
}

const postJson = (path, body) =>
  request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

const postFile = (path, file) => {
  const form = new FormData()
  form.append('file', file)
  return request(path, { method: 'POST', body: form })
}

export const api = {
  systemInfo: (warm = false) => request(`/api/system/info?warm=${warm}`),
  health: () => request('/api/health'),

  uploadMap: (file) => postFile('/api/map/upload', file),
  mapStatus: (mapId) => request(`/api/map/${mapId}`),
  uploadDrone: (file) => postFile('/api/drone/upload', file),
  uploadPlan: (file) => postFile('/api/plan/upload', file),

  setGeoreference: (body) => postJson('/api/georeference', body),
  clearGeoreference: (mapId) =>
    request(`/api/georeference/${mapId}`, { method: 'DELETE' }),

  localize: (body) => postJson('/api/localize', body),
  jobStatus: (jobId) => request(`/api/process/${jobId}`),
  jobResult: (jobId) => request(`/api/result/${jobId}`),
  candidates: (jobId) => request(`/api/candidates/${jobId}`),

  batchTest: (body) => postJson('/api/dev/batch', body),
}

export { ApiError }
