"use client";

import { useEffect, useRef } from "react";

/**
 * Drifting grid behind the app. Canvas rather than CSS because a DOM grid
 * at this density costs a layout pass per frame.
 */
export function BackgroundSquares({
  speed = 0.35,
  size = 40,
  color = "#e2a52c",
}: {
  speed?: number;
  size?: number;
  color?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    let frame = 0;
    let offset = 0;
    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    // Respect the OS setting instead of animating into someone's migraine.
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!still) offset = (offset + speed) % size;
      const start = -size + offset;
      ctx.strokeStyle = color;
      ctx.lineWidth = 0.5;
      ctx.globalAlpha = 0.13;
      ctx.beginPath();
      for (let x = start; x < canvas.width + size; x += size) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
      }
      for (let y = start; y < canvas.height + size; y += size) {
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
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
  }, [speed, size, color]);

  return <canvas ref={ref} className="fixed inset-0 -z-10 bg-black" />;
}
