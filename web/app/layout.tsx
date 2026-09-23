import type { Metadata } from "next";
import { BackgroundSquares } from "@/components/BackgroundSquares";
import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aletheia — Fact Knowledge Layer",
  description:
    "Extracts facts from PDFs, grounds each in source evidence, and reconciles them across documents.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <BackgroundSquares />
        <Nav />
        <main className="relative z-10 mx-auto max-w-[1480px] px-6 pb-24">{children}</main>
      </body>
    </html>
  );
}
