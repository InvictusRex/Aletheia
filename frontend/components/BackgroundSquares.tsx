"use client";

import { useEffect, useRef } from "react";

/**
 * Drifting grid behind the app.
 *
 * Sits at z-0 rather than a negative z-index: a negative one paints
 * behind the root background, which is opaque black, so the grid was
 * invisible. Content stacks above it instead.
 */
export function BackgroundSquares({
  speed = 0.35,
  size = 40,
  color = "#e2a52c",
  opacity = 0.22,
}: {
  speed?: number;
  size?: number;
  color?: string;
  opacity?: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    let frame = 0;
    let offset = 0;

    // Match the backing store to the device pixel ratio, otherwise a
    // 0.5px line on a HiDPI screen renders as a blurry smear.
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(window.innerWidth * dpr);
      canvas.height = Math.floor(window.innerHeight * dpr);
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    // Respect the OS setting instead of animating into someone's migraine.
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const draw = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      ctx.clearRect(0, 0, w, h);
      if (!still) offset = (offset + speed) % size;

      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = opacity;
      ctx.beginPath();
      for (let x = -size + offset; x < w + size; x += size) {
        ctx.moveTo(Math.round(x) + 0.5, 0);
        ctx.lineTo(Math.round(x) + 0.5, h);
      }
      for (let y = -size + offset; y < h + size; y += size) {
        ctx.moveTo(0, Math.round(y) + 0.5);
        ctx.lineTo(w, Math.round(y) + 0.5);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      if (!still) frame = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(frame);
    };
  }, [speed, size, color, opacity]);

  return (
    <canvas
      ref={ref}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-0"
    />
  );
}
