"use client";

import { Sparkles, Table2, UploadCloud, TrendingUp, Check } from "lucide-react";

const bars = [42, 68, 55, 88, 74, 96, 61, 80];
const line = "M0,58 L36,44 L72,50 L108,30 L144,36 L180,18 L216,26 L252,8";

export function MiniChart({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 252 64" className={className} preserveAspectRatio="none">
      <defs>
        <linearGradient id="dp-line" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="var(--blue)" />
          <stop offset="100%" stopColor="var(--cyan)" />
        </linearGradient>
        <linearGradient id="dp-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--blue)" stopOpacity="0.45" />
          <stop offset="100%" stopColor="var(--blue)" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${line} L252,64 L0,64 Z`} fill="url(#dp-fill)" />
      <path d={line} fill="none" stroke="url(#dp-line)" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function DashboardMockup() {
  return (
    <div className="glass-card glow-ring overflow-hidden rounded-2xl">
      {/* window bar */}
      <div className="flex items-center gap-2 border-b border-border/60 bg-deep/40 px-4 py-3">
        <span className="size-2.5 rounded-full bg-destructive/70" />
        <span className="size-2.5 rounded-full bg-accent/60" />
        <span className="size-2.5 rounded-full bg-primary/70" />
        <div className="ml-3 flex items-center gap-2 rounded-md bg-surface-2/60 px-3 py-1 text-[11px] text-muted-foreground">
          <Table2 className="size-3" /> sales_q3_2026.xlsx
        </div>
        <span className="ml-auto hidden items-center gap-1.5 text-[11px] text-accent sm:flex">
          <span className="size-1.5 animate-pulse-soft rounded-full bg-accent" /> agents active
        </span>
      </div>

      <div className="grid gap-4 p-4 sm:p-5 lg:grid-cols-[1.6fr_1fr]">
        <div className="space-y-4">
          {/* prompt */}
          <div className="rounded-xl border border-primary/25 bg-surface-2/40 p-3">
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
              <UploadCloud className="size-3.5 text-accent" /> Dataset uploaded · 12,480 rows · 9
              columns
            </div>
            <div className="mt-2.5 flex items-start gap-2">
              <Sparkles className="mt-0.5 size-4 shrink-0 text-accent" />
              <p className="text-sm leading-snug text-foreground">
                “Analyze the sales data and show the top-performing regions.”
              </p>
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {["Understand", "Plan", "Generate", "Execute", "Validate"].map((s, i) => (
                <span
                  key={s}
                  className="rounded-full border border-primary/25 bg-primary/10 px-2 py-0.5 text-[10px] text-foreground/80"
                  style={{ animation: `dp-pulse-soft 3.2s ease-in-out ${i * 0.25}s infinite` }}
                >
                  {s}
                </span>
              ))}
            </div>
          </div>

          {/* charts */}
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-border/60 bg-deep/40 p-3">
              <p className="mb-2 text-[11px] text-muted-foreground">Revenue by region</p>
              <div className="flex h-24 items-end gap-1.5">
                {bars.map((h, i) => (
                  <div
                    key={i}
                    className="flex-1 origin-bottom rounded-t-sm bg-gradient-to-t from-primary/40 to-accent/80"
                    style={{ height: `${h}%`, animation: `dp-rise .9s ease-out ${i * 0.07}s both` }}
                  />
                ))}
              </div>
            </div>
            <div className="rounded-xl border border-border/60 bg-deep/40 p-3">
              <p className="mb-2 text-[11px] text-muted-foreground">Monthly trend</p>
              <MiniChart className="h-24 w-full" />
            </div>
          </div>

          {/* table */}
          <div className="overflow-hidden rounded-xl border border-border/60 bg-deep/40">
            <div className="grid grid-cols-4 gap-2 border-b border-border/60 px-3 py-2 text-[10px] uppercase tracking-wider text-muted-foreground">
              <span>Region</span>
              <span>Units</span>
              <span>Revenue</span>
              <span>Growth</span>
            </div>
            {[
              ["North", "4,182", "$1.24M", "+12.4%"],
              ["West", "3,910", "$1.08M", "+9.1%"],
              ["Central", "2,764", "$842K", "+4.6%"],
            ].map((r) => (
              <div
                key={r[0]}
                className="grid grid-cols-4 gap-2 px-3 py-2 text-[11px] text-foreground/85 odd:bg-surface-2/25"
              >
                <span>{r[0]}</span>
                <span className="font-mono">{r[1]}</span>
                <span className="font-mono">{r[2]}</span>
                <span className="font-mono text-accent">{r[3]}</span>
              </div>
            ))}
          </div>
        </div>

        {/* side */}
        <div className="space-y-3">
          {[
            ["Total revenue", "$3.42M"],
            ["Top region", "North"],
            ["Rows cleaned", "1,207"],
          ].map(([k, v]) => (
            <div key={k} className="rounded-xl border border-border/60 bg-surface-2/35 p-3">
              <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{k}</p>
              <p className="mt-1 text-lg font-semibold text-foreground">{v}</p>
            </div>
          ))}
          <div className="rounded-xl border border-accent/25 bg-accent/10 p-3">
            <p className="flex items-center gap-1.5 text-[11px] font-medium text-accent">
              <TrendingUp className="size-3.5" /> AI insight
            </p>
            <p className="mt-1.5 text-[12px] leading-relaxed text-foreground/85">
              North leads revenue with the strongest quarter-over-quarter growth; Central lags on
              units despite steady margins.
            </p>
            <p className="mt-2 flex items-center gap-1 text-[10px] text-muted-foreground">
              <Check className="size-3 text-accent" /> Result validated
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
