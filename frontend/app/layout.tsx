import type { Metadata } from "next";
import { BackgroundSquares } from "@/components/BackgroundSquares";
import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aletheia",
  description:
    "Extracts facts from PDFs, grounds each in source evidence, and reconciles them across documents.",
  icons: { icon: "/eye.svg" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <BackgroundSquares />
        <Nav />
        <main className="relative z-10 mx-auto max-w-[90rem] px-4 pt-16">
          {children}
        </main>
      </body>
    </html>
  );
}
