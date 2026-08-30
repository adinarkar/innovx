/** Formatting helpers shared across the dashboard and analysis views. */

export const percent = (value, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '--'
    : `${(Number(value) * 100).toFixed(digits)}%`

export const number = (value, digits = 0) =>
  value === null || value === undefined || Number.isNaN(value)
    ? '--'
    : Number(value).toFixed(digits)

export const px = (value, digits = 1) =>
  value === null || value === undefined ? '--' : `${Number(value).toFixed(digits)} px`

export const bytes = (size) => {
  if (!size && size !== 0) return '--'
  const units = ['B', 'KB', 'MB', 'GB']
  let v = Number(size)
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

export const seconds = (value) =>
  value === null || value === undefined ? '--' : `${Number(value).toFixed(2)}s`

export const coord = (value, digits = 6) =>
  value === null || value === undefined ? '--' : Number(value).toFixed(digits)

/** Reduce a ratio to a friendly aspect label, e.g. 1.78 -> "16:9". */
export const aspect = (ratio) => {
  if (!ratio) return '--'
  const known = [
    [1, '1:1'],
    [4 / 3, '4:3'],
    [3 / 2, '3:2'],
    [16 / 9, '16:9'],
    [3 / 4, '3:4'],
    [2 / 3, '2:3'],
    [9 / 16, '9:16'],
  ]
  const hit = known.find(([v]) => Math.abs(v - ratio) < 0.02)
  return hit ? hit[1] : `${Number(ratio).toFixed(2)}:1`
}

/** Machine rejection codes -> presenter-friendly wording. */
export const rejectionLabel = (code) =>
  ({
    too_few_matches: 'Too few correspondences to solve a homography',
    homography_failed: 'RANSAC could not fit a homography',
    insufficient_inliers: 'Not enough inliers survived RANSAC',
    non_finite_projection: 'Projection produced invalid coordinates',
    non_convex_projection: 'Projected frame is not a convex quadrilateral',
    degenerate_scale: 'Transform collapses the frame',
    implausible_scale_ratio: 'Implausible area change between frame and map',
    excessive_shear: 'Excessive shear for a planar aerial view',
    degenerate_edges: 'Projected frame has degenerate edges',
    extreme_perspective: 'Extreme perspective distortion',
    below_min_inliers: 'Below the minimum inlier count',
    below_min_inlier_ratio: 'Below the minimum inlier ratio',
    reprojection_error_too_high: 'Reprojection error above threshold',
    features_too_concentrated: 'Inliers concentrated in too few grid cells',
    no_valid_homography: 'No valid homography',
  })[code] || (code ? code.replace(/_/g, ' ') : null)

export const STATUS_META = {
  MATCH_FOUND: {
    label: 'Match Found',
    tone: 'ok',
    detail: 'Geometrically verified against the reference map.',
  },
  LOW_CONFIDENCE: {
    label: 'Low Confidence',
    tone: 'warn',
    detail: 'Visual localization unreliable - indicative position only.',
  },
  AMBIGUOUS: {
    label: 'Ambiguous',
    tone: 'warn',
    detail: 'Several map regions explain this frame equally well.',
  },
  NO_MATCH: {
    label: 'No Match',
    tone: 'bad',
    detail: 'The frame does not correspond to any region of this map.',
  },
}
