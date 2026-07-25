# Frontend Changes: Dark/Light Theme Toggle

## Summary

Added a dark/light theme toggle button to the Course Materials Assistant UI. The app defaults to the existing dark theme; users can switch to a light theme via a fixed icon button in the top-right corner. The choice persists across sessions via `localStorage` and falls back to the OS-level `prefers-color-scheme` on first visit.

## Files Changed

### `frontend/index.html`
- Added an inline `<script>` in `<head>` that applies `data-theme="light"` to `<html>` *before first paint*, based on `localStorage.theme` (falling back to `prefers-color-scheme: light`), to avoid a flash of the wrong theme on load.
- Added a `#themeToggle` button (fixed top-right) containing sun/moon SVG icons, right after `<body>` opens.
- Bumped `style.css`/`script.js` cache-busting query params from `v=10` to `v=11`.

### `frontend/style.css`
- Added a `:root[data-theme="light"]` block overriding the theme CSS variables (`--background`, `--surface`, `--surface-hover`, `--text-primary`, `--text-secondary`, `--border-color`, `--assistant-message`, `--shadow`, `--focus-ring`, `--welcome-bg`, `--welcome-border`) with light-mode equivalents. `--primary-color`/`--primary-hover` (brand blue) are shared across both themes.
- Added `background-color`/`color`/`border-color` transitions (`body` and `.container *`) so switching themes animates smoothly instead of snapping.
- Added `.theme-toggle` button styles (circular, fixed position, hover/focus states) and `.theme-icon` sun/moon cross-fade + rotate animation driven by the `data-theme` attribute. Reduced the button's size slightly at the existing `max-width: 768px` breakpoint.
- Added light-theme-specific color overrides for `.source-link`, `.error-message`, and `.success-message` so their text keeps sufficient contrast against a white background (their existing dark-mode colors were tuned for a dark background).
- Fixed a pre-existing dead CSS variable reference: `.message-content blockquote` referenced `var(--primary)`, which was never defined (only `--primary-color` exists), silently dropping the blockquote's left border. Changed to `var(--primary-color)`.

### `frontend/script.js`
- Added `themeToggle` to the tracked DOM elements and wired its `click` handler in `setupEventListeners()`.
- Added `initializeTheme()` (syncs the toggle button's `aria-label`/`title` with whatever theme the inline head script already applied), `updateThemeToggleLabel()`, and `toggleTheme()` (flips the `data-theme` attribute, persists the choice to `localStorage`, updates the label). Called `initializeTheme()` from `DOMContentLoaded`.

## Behavior

- Default: dark theme (unchanged existing look).
- First visit with no saved preference: follows the OS `prefers-color-scheme`.
- Clicking the toggle switches themes immediately, with a smooth color transition, and persists the choice for future visits.
- Toggle is keyboard-accessible (native `<button>`, visible focus ring) and has a dynamic `aria-label`/`title` describing the action it will perform.

## Not verified

Could not visually test in a running browser — the local dev server on port 8000 was serving a different git worktree (`main`), not this `ui_feature` worktree, and per project instructions the dev server should not be started here. Changes were verified via code review only.
