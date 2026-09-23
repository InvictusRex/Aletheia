/** Animated "..." for a value that is still being fetched. */
export function Dots() {
  return (
    <span className="dots text-muted" aria-label="loading">
      <span>.</span>
      <span>.</span>
      <span>.</span>
    </span>
  );
}
