"use client";

import {
  BarChart3,
  LineChart,
  PieChart,
  Table,
  Database,
  Network,
  ScatterChart,
  Sigma,
} from "lucide-react";
import { Parallax } from "./parallax";

export function GridBackdrop() {
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      <Parallax speed={0.28} className="absolute inset-0">
        <div className="grid-pattern absolute inset-[-20%] opacity-[0.5] [mask-image:radial-gradient(75%_65%_at_50%_25%,black,transparent)]" />
      </Parallax>
      <div
        className="absolute inset-0"
        style={{ background: "var(--gradient-hero)", opacity: 0.9 }}
      />
    </div>
  );
}

const ICONS = [BarChart3, LineChart, PieChart, Table, Database, Network, ScatterChart, Sigma];

type Spec = { x: string; y: string; size: number; speed: number; icon: number; hideSm?: boolean };

const DEFAULT_SPECS: Spec[] = [
  { x: "6%", y: "18%", size: 26, speed: 0.5, icon: 0 },
  { x: "88%", y: "12%", size: 22, speed: 0.75, icon: 1, hideSm: true },
  { x: "80%", y: "66%", size: 30, speed: 0.42, icon: 3 },
  { x: "68%", y: "30%", size: 16, speed: 0.95, icon: 6, hideSm: true },
];

export function FloatingIcons({ specs = DEFAULT_SPECS }: { specs?: Spec[] }) {
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      {specs.map((s, i) => {
        const Icon = ICONS[s.icon % ICONS.length]!;
        return (
          <Parallax
            key={i}
            speed={s.speed}
            className={`absolute ${s.hideSm ? "hidden md:block" : ""}`}
            style={{ left: s.x, top: s.y }}
          >
            <div
              className="animate-float rounded-xl border border-primary/20 bg-surface/40 p-2 backdrop-blur-sm"
              style={{
                animationDelay: `${i * 0.8}s`,
                boxShadow: "0 0 30px -10px color-mix(in oklab, var(--blue) 60%, transparent)",
              }}
            >
              <Icon
                style={{ width: s.size, height: s.size }}
                className="text-accent/70"
                strokeWidth={1.4}
              />
            </div>
          </Parallax>
        );
      })}
    </div>
  );
}

export function Particles({ count = 15 }: { count?: number }) {
  const dots = Array.from({ length: count }, (_, i) => ({
    left: `${(i * 37) % 100}%`,
    top: `${(i * 61) % 100}%`,
    d: 3 + (i % 5),
    delay: (i % 7) * 0.6,
  }));
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      {dots.map((d, i) => (
        <span
          key={i}
          className="absolute rounded-full bg-accent/50 animate-pulse-soft"
          style={{
            left: d.left,
            top: d.top,
            width: d.d,
            height: d.d,
            animationDelay: `${d.delay}s`,
            boxShadow: "0 0 10px color-mix(in oklab, var(--cyan) 70%, transparent)",
          }}
        />
      ))}
    </div>
  );
}

/** Background depth: large blurred navy-blue radial glows moving at slow speeds. */
export function Atmosphere() {
  const layers = [
    { speed: 0.18, x: "18%", y: "10%", size: 620, tint: "var(--blue)", opacity: 0.22 },
    { speed: 0.32, x: "82%", y: "38%", size: 520, tint: "var(--cyan)", opacity: 0.14 },
    { speed: 0.5, x: "45%", y: "88%", size: 460, tint: "var(--blue)", opacity: 0.16 },
  ];
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      {layers.map((l, i) => (
        <Parallax key={i} speed={l.speed} className="absolute" style={{ left: l.x, top: l.y }}>
          <div
            className="-translate-x-1/2 -translate-y-1/2 rounded-full blur-3xl"
            style={{
              width: l.size,
              height: l.size,
              opacity: l.opacity,
              background: `radial-gradient(circle, ${l.tint}, transparent 70%)`,
            }}
          />
        </Parallax>
      ))}
    </div>
  );
}

/** Middle ground: faint spreadsheet cells, data-flow lines and nodes. */
export function DataLayer() {
  const cells = [
    { x: "4%", y: "26%", speed: 0.55, rows: 3, cols: 3 },
    { x: "86%", y: "62%", speed: 0.7, rows: 2, cols: 4, hideSm: true },
  ];
  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      <Parallax speed={0.42} className="absolute inset-x-0 top-1/3">
        <svg viewBox="0 0 1200 120" preserveAspectRatio="none" className="h-24 w-full opacity-40">
          <path
            d="M0,80 C220,20 380,110 600,60 C820,10 980,100 1200,45"
            fill="none"
            stroke="var(--blue)"
            strokeOpacity="0.35"
            strokeWidth="1.2"
            className="animate-dash"
          />
          <path
            d="M0,30 C260,90 420,10 640,70 C860,120 1020,20 1200,90"
            fill="none"
            stroke="var(--cyan)"
            strokeOpacity="0.2"
            strokeWidth="1"
            className="animate-dash"
          />
        </svg>
      </Parallax>
      {cells.map((c, i) => (
        <Parallax
          key={i}
          speed={c.speed}
          className={`absolute ${c.hideSm ? "hidden md:block" : ""}`}
          style={{ left: c.x, top: c.y }}
        >
          <div
            className="grid gap-px rounded-lg border border-primary/15 bg-surface/20 p-px backdrop-blur-[2px]"
            style={{ gridTemplateColumns: `repeat(${c.cols}, 26px)` }}
          >
            {Array.from({ length: c.rows * c.cols }, (_, k) => (
              <span
                key={k}
                className="h-5 rounded-[3px] bg-primary/10"
                style={{ animation: `dp-pulse-soft ${5 + (k % 4)}s ease-in-out ${k * 0.2}s infinite` }}
              />
            ))}
          </div>
        </Parallax>
      ))}
    </div>
  );
}
