/**
 * The browser reaches the API at the same origin as the page, under /api,
 * and this rewrite forwards that to the backend. Behind a tunnel that
 * means one hostname and no CORS at all; locally it changes nothing,
 * because NEXT_PUBLIC_ALETHEIA_API_URL still points straight at :8000.
 *
 * Rewrites are resolved when the production build is made, so the
 * destination is a build argument rather than a runtime variable.
 */
const API_ORIGIN = process.env.ALETHEIA_API_URL ?? "http://localhost:8000";

/** @type {import('next').NextConfig} */
export default {
  images: { unoptimized: true },
  env: { ALETHEIA_API_URL: API_ORIGIN },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_ORIGIN}/:path*` }];
  },
};
