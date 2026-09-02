"use client";

import Link from "next/link";
import { ArrowLeft, PlayCircle } from "lucide-react";
import "./home.css";

export function TutorialsView() {
  return (
    <div className="datapilot-home relative flex min-h-screen items-center justify-center px-4">
      <div className="bg-grid-texture pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_at_center,black,transparent_75%)]" />
      <div className="relative w-full max-w-lg rounded-2xl border border-border bg-card p-10 text-center shadow-card">
        <span className="mx-auto flex size-14 items-center justify-center rounded-2xl bg-emerald-soft text-primary ring-1 ring-primary/25">
          <PlayCircle className="size-7" strokeWidth={1.8} />
        </span>
        <h1 className="mt-6 text-2xl font-semibold tracking-tight text-foreground">
          Tutorials
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          Guided video walkthroughs for DataPilot will live here — from your
          first upload to interpreting AI-generated insights.
        </p>
        <Link
          href="/home"
          className="mt-8 inline-flex items-center gap-1.5 rounded-lg border border-border bg-secondary px-4 py-2.5 text-sm font-medium text-secondary-foreground transition-colors hover:border-primary/40 hover:text-primary"
        >
          <ArrowLeft className="size-4" />
          Back to Home
        </Link>
      </div>
    </div>
  );
}
