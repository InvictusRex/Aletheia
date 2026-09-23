/**
 * Aletheia: the Greek word for truth as *disclosure*, what is no longer
 * hidden. An open eye is the mark.
 *
 * The lid curves are cubic Béziers whose control points sit 70 units off
 * the centre line, so the lid peaks around y=11 and y=117 on a 240-wide
 * eye. Pulling those controls closer to y=64 narrows the opening; pushing
 * them further apart widens it. The viewBox starts at -28 to leave the
 * lashes room above the lid.
 */
export function EyeLogo({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 -28 280 156"
      className={className}
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M20 64 C60 -6, 220 -6, 260 64 C220 134, 60 134, 20 64 Z"
        stroke="currentColor"
        strokeWidth="8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="140" cy="64" r="27" fill="currentColor" />
      {/*
        Lashes are mirrored about the eye's centre (x=140) and each starts
        just ABOVE the lid at its own x, because the lid slopes: at x=58 it
        sits at y≈30, at x=126 only y≈12. A lash starting below its local
        lid height lands inside the eye.
      */}
      <path d="M58 25 L44 2" stroke="currentColor" strokeWidth="7" strokeLinecap="round" />
      <path d="M96 12 L88 -12" stroke="currentColor" strokeWidth="7" strokeLinecap="round" />
      <path d="M126 7 L124 -19" stroke="currentColor" strokeWidth="7" strokeLinecap="round" />
      <path d="M154 7 L156 -19" stroke="currentColor" strokeWidth="7" strokeLinecap="round" />
      <path d="M184 12 L192 -12" stroke="currentColor" strokeWidth="7" strokeLinecap="round" />
      <path d="M222 25 L236 2" stroke="currentColor" strokeWidth="7" strokeLinecap="round" />
    </svg>
  );
}
