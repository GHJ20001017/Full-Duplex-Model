// Orb placement policy: where the orb lives on screen, and remembering where
// the user dragged it.
//
// The module stays free of any electron import: displays, the window size and
// the persistence are injected, which keeps the policy testable. Display shape
// follows electron's: { id, workArea: { x, y, width, height } }. Persisted
// state shape: { x, y, displayId }.

export const ORB_PLACEMENT_MARGIN = 24

function validSavedState(state) {
  return (
    state
    && Number.isFinite(state.x)
    && Number.isFinite(state.y)
  )
}

export function createOrbPlacement({
  getDisplays,
  orbSize,
  margin = ORB_PLACEMENT_MARGIN,
  loadState = () => null,
  saveState = () => {},
} = {}) {
  if (typeof getDisplays !== 'function') {
    throw new Error('createOrbPlacement: getDisplays is required')
  }
  if (!orbSize?.width || !orbSize?.height) {
    throw new Error('createOrbPlacement: orbSize is required')
  }

  const displays = () => (getDisplays() || [])
    .filter(display => display && display.workArea)

  // The default anchor is the display's work area, top-right with a margin:
  // workArea already excludes the menu bar and the Dock, so the orb never
  // lands under system chrome, and no display is hard-coded.
  const defaultPosition = display => {
    const { workArea } = display
    return {
      x: workArea.x + workArea.width - orbSize.width - margin,
      y: workArea.y + margin,
    }
  }

  const clampToDisplay = (position, display) => {
    const { workArea } = display
    const maxX = workArea.x + Math.max(0, workArea.width - orbSize.width)
    const maxY = workArea.y + Math.max(0, workArea.height - orbSize.height)
    return {
      x: Math.min(Math.max(position.x, workArea.x), maxX),
      y: Math.min(Math.max(position.y, workArea.y), maxY),
    }
  }

  // The display hosting a position: the one whose work area contains the orb
  // centre, else the nearest one by centre distance — a display the saved
  // position referenced may have been unplugged.
  const displayForPosition = position => {
    const list = displays()
    if (!list.length) return null
    const cx = position.x + orbSize.width / 2
    const cy = position.y + orbSize.height / 2
    const containing = list.find(display => {
      const { workArea } = display
      return cx >= workArea.x && cx < workArea.x + workArea.width
        && cy >= workArea.y && cy < workArea.y + workArea.height
    })
    if (containing) return containing
    let nearest = list[0]
    let bestDistance = Number.POSITIVE_INFINITY
    for (const display of list) {
      const { workArea } = display
      const dx = workArea.x + workArea.width / 2 - cx
      const dy = workArea.y + workArea.height / 2 - cy
      const distance = dx * dx + dy * dy
      if (distance < bestDistance) {
        bestDistance = distance
        nearest = display
      }
    }
    return nearest
  }

  return {
    defaultPosition,
    clampToDisplay,
    displayForPosition,

    // Where a new window should appear: the position the user last dragged
    // the orb to, clamped into whichever display now hosts it; the default
    // top-right anchor of the primary display otherwise.
    initialPosition() {
      const list = displays()
      if (!list.length) return { x: margin, y: margin }
      let saved = null
      try {
        saved = loadState()
      } catch {
        // A corrupt state file falls back to the default anchor.
      }
      if (validSavedState(saved)) {
        const display = displayForPosition(saved)
        if (display) return clampToDisplay(saved, display)
      }
      return defaultPosition(list[0])
    },

    // Persists where the user left the orb. Failures stay silent: losing a
    // position record must never break the drag itself.
    recordPosition(position) {
      if (!validSavedState(position)) return false
      const display = displayForPosition(position)
      try {
        saveState({
          x: Math.round(position.x),
          y: Math.round(position.y),
          displayId: display?.id ?? null,
        })
        return true
      } catch {
        return false
      }
    },
  }
}
