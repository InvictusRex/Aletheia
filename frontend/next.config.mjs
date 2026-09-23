/** @type {import('next').NextConfig} */
export default {
  images: { unoptimized: true },
  // The browser talks to the API directly; this is only the server-side default.
  env: { ALETHEIA_API_URL: process.env.ALETHEIA_API_URL ?? "http://localhost:8000" },
};
