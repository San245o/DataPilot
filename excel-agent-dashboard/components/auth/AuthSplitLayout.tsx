"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { ArrowLeft, Compass, ShieldCheck } from "lucide-react";
import { DataPilotAnimation } from "./DataPilotAnimation";

interface AuthSplitLayoutProps {
  children: ReactNode;
}

export function AuthSplitLayout({ children }: AuthSplitLayoutProps) {
  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100 lg:flex-row">
      {/* Left Brand Panel - Dark Slate with Emerald & Cyan accents */}
      <aside className="relative flex min-h-[420px] w-full flex-col justify-between overflow-hidden border-b border-slate-800/80 bg-slate-900/40 px-6 py-8 sm:px-10 lg:min-h-screen lg:w-[46%] lg:border-b-0 lg:border-r lg:px-12 lg:py-12">
        {/* Subtle grid pattern */}
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.03] [background-image:linear-gradient(rgba(255,255,255,0.4)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.4)_1px,transparent_1px)] [background-size:32px_32px]"
          aria-hidden="true"
        />

        <div className="relative z-10">
          {/* Logo */}
          <Link href="/" className="inline-flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-xl border border-slate-800 bg-slate-900 text-emerald-400 shadow-sm">
              <Compass className="size-5" />
            </span>
            <span className="text-lg font-bold tracking-tight text-slate-100">
              DataPilot
            </span>
          </Link>

          {/* Value Prop */}
          <div className="mt-10 max-w-lg sm:mt-14">
            <div className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/90 px-3 py-1 text-xs font-medium text-slate-300">
              <span className="size-1.5 rounded-full bg-emerald-400" />
              <span>Next-Gen Spreadsheet Intelligence</span>
            </div>

            <h2 className="mt-5 text-2xl font-bold tracking-tight text-slate-100 sm:text-3xl lg:text-4xl">
              Your data has answers.
              <br />
              <span className="text-emerald-400">DataPilot</span> helps you find them.
            </h2>

            <p className="mt-3 text-xs leading-relaxed text-slate-400 sm:text-sm">
              Clean messy sheets, execute Python models in an isolated sandbox, and generate
              multi-chart reports with conversational AI.
            </p>
          </div>

          {/* Animation Container */}
          <div className="mt-8 w-full max-w-md">
            <DataPilotAnimation className="h-44 w-full sm:h-52 lg:h-64" />
          </div>
        </div>

        {/* Footer Note */}
        <div className="relative z-10 mt-6 hidden items-center gap-2 text-xs text-slate-500 lg:flex">
          <ShieldCheck className="size-4 text-emerald-400" />
          <span>Sandboxed Python runtime · End-to-end data integrity</span>
        </div>
      </aside>

      {/* Right Form Panel - Sleek Dark Slate Matching Dashboard */}
      <main className="relative flex min-h-[calc(100vh-420px)] flex-1 items-center justify-center bg-slate-950 px-6 py-12 text-slate-100 sm:px-10 lg:min-h-screen lg:px-14 lg:py-16">
        <Link
          href="/"
          className="absolute left-6 top-6 inline-flex items-center gap-2 rounded-lg border border-slate-800/80 bg-slate-900/60 px-3 py-1.5 text-xs font-medium text-slate-400 transition-colors hover:border-slate-700 hover:text-white sm:left-10 sm:top-8"
        >
          <ArrowLeft className="size-3.5" />
          <span>Back to home</span>
        </Link>

        <div className="w-full max-w-md pt-8 sm:pt-0">
          {children}
        </div>
      </main>
    </div>
  );
}

export default AuthSplitLayout;
