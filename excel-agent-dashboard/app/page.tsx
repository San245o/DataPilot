"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import {
  Activity,
  ArrowRight,
  BarChart3,
  Brain,
  CheckCircle2,
  Code2,
  Compass,
  Database,
  FileSpreadsheet,
  FileText,
  History,
  Layers,
  Lock,
  Menu,
  MessageSquareText,
  Network,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Table2,
  Terminal,
  UploadCloud,
  Wand2,
  X,
} from "lucide-react"
import { DashboardMockup } from "@/components/dashboard-mockup"
import {
  Parallax,
  Reveal,
  TiltCard,
  useScrollProgress,
} from "@/components/parallax"

const NAV_LINKS = [
  { label: "Features", href: "#features" },
  { label: "Architecture", href: "#architecture" },
  { label: "Workflow", href: "#workflow" },
  { label: "Capabilities", href: "#capabilities" },
]

const TYPEWRITER_PHRASES = [
  "Multi-Chart Reports",
  "Sandboxed Python Models",
  "Conversational Insights",
  "Automated Data Cleaning",
  "Publication-Ready PDF Reports",
]

function ScrollProgressBar() {
  const p = useScrollProgress()
  return (
    <div className="fixed inset-x-0 top-0 z-[60] h-0.5 bg-transparent pointer-events-none">
      <div
        className="h-full bg-gradient-to-r from-emerald-500 via-teal-400 to-sky-400 transition-[width] duration-150"
        style={{ width: `${p * 100}%` }}
      />
    </div>
  )
}

function ShimmerDotBackground() {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="shimmer-dots-bg absolute inset-0 opacity-45 [mask-image:radial-gradient(ellipse_75%_60%_at_50%_25%,black_30%,transparent_100%)]" />
    </div>
  )
}

function TypewriterText({ words }: { words: string[] }) {
  const [wordIndex, setWordIndex] = useState(0)
  const [text, setText] = useState("")
  const [isDeleting, setIsDeleting] = useState(false)

  useEffect(() => {
    const currentWord = words[wordIndex % words.length]
    let timeout: NodeJS.Timeout

    if (!isDeleting) {
      if (text.length < currentWord.length) {
        timeout = setTimeout(() => {
          setText(currentWord.slice(0, text.length + 1))
        }, 65)
      } else {
        timeout = setTimeout(() => {
          setIsDeleting(true)
        }, 2200)
      }
    } else {
      if (text.length > 0) {
        timeout = setTimeout(() => {
          setText(currentWord.slice(0, text.length - 1))
        }, 35)
      } else {
        setIsDeleting(false)
        setWordIndex((prev) => prev + 1)
      }
    }

    return () => clearTimeout(timeout)
  }, [text, isDeleting, wordIndex, words])

  return (
    <span className="inline-flex items-center text-emerald-400">
      <span>{text}</span>
      <span
        className="ml-1 inline-block h-[0.85em] w-[2.5px] bg-emerald-400 align-middle"
        style={{ animation: "cursor-blink 0.9s steps(2, start) infinite" }}
        aria-hidden="true"
      />
    </span>
  )
}

function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2.5">
      <span className="flex size-8 items-center justify-center rounded-lg border border-slate-800 bg-slate-900 text-emerald-400 shadow-sm">
        <Compass className="size-4" />
      </span>
      <span className="text-base font-bold tracking-tight text-slate-100">
        DataPilot
      </span>
    </Link>
  )
}

function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    onScroll()
    window.addEventListener("scroll", onScroll, { passive: true })
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 transition-all duration-200 ${
        scrolled
          ? "border-b border-slate-800/80 bg-slate-950/90 backdrop-blur-md"
          : "border-b border-transparent bg-transparent"
      }`}
    >
      <nav className="mx-auto flex h-14 max-w-6xl items-center justify-between px-5">
        <Logo />

        <ul className="hidden items-center gap-6 md:flex">
          {NAV_LINKS.map((link) => (
            <li key={link.label}>
              <a
                href={link.href}
                className="text-xs font-medium text-slate-400 transition-colors hover:text-slate-100"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>

        <div className="hidden items-center gap-3 sm:flex">
          <Link
            href="/login"
            className="rounded-lg px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:bg-slate-900 hover:text-white"
          >
            Sign In
          </Link>
          <Link
            href="/dashboard"
            className="flex items-center gap-1.5 rounded-lg bg-emerald-500 px-3.5 py-1.5 text-xs font-semibold text-slate-950 transition-colors hover:bg-emerald-400 shadow-sm"
          >
            Launch Dashboard
            <ArrowRight className="size-3.5" />
          </Link>
        </div>

        <button
          type="button"
          className="text-slate-400 hover:text-white md:hidden"
          onClick={() => setMobileOpen((o) => !o)}
          aria-label="Toggle navigation"
        >
          {mobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </nav>

      {mobileOpen && (
        <div className="border-b border-slate-800 bg-slate-950 px-5 pb-5 pt-2 md:hidden">
          <ul className="space-y-1">
            {NAV_LINKS.map((link) => (
              <li key={link.label}>
                <a
                  href={link.href}
                  onClick={() => setMobileOpen(false)}
                  className="block rounded-lg px-3 py-2 text-sm text-slate-300 hover:bg-slate-900 hover:text-white"
                >
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <Link
              href="/login"
              onClick={() => setMobileOpen(false)}
              className="rounded-lg border border-slate-800 bg-slate-900 py-2 text-center text-xs font-medium text-slate-300"
            >
              Sign In
            </Link>
            <Link
              href="/dashboard"
              onClick={() => setMobileOpen(false)}
              className="rounded-lg bg-emerald-500 py-2 text-center text-xs font-semibold text-slate-950"
            >
              Launch Dashboard
            </Link>
          </div>
        </div>
      )}
    </header>
  )
}

function Hero() {
  return (
    <section className="relative overflow-hidden pt-28 pb-16 sm:pt-36 sm:pb-24 border-b border-slate-800/60">
      <ShimmerDotBackground />

      <div className="relative mx-auto max-w-6xl px-5 text-center">
        <Reveal>
          <div className="inline-flex items-center gap-2 rounded-full border border-slate-800/80 bg-slate-900/90 px-3.5 py-1 text-xs font-medium text-slate-300 backdrop-blur-sm">
            <span className="size-1.5 rounded-full bg-emerald-400" />
            <span>Next-Gen Spreadsheet Intelligence</span>
          </div>
        </Reveal>

        <Reveal delay={80}>
          <h1 className="mx-auto mt-6 max-w-4xl text-balance text-3xl font-bold tracking-tight text-slate-100 sm:text-5xl lg:text-6xl">
            Turn Spreadsheets Into
            <br />
            <TypewriterText words={TYPEWRITER_PHRASES} />
          </h1>
        </Reveal>

        <Reveal delay={160}>
          <p className="mx-auto mt-5 max-w-2xl text-balance text-xs leading-relaxed text-slate-400 sm:text-base">
            DataPilot pairs conversational AI with a secure, sandboxed Python environment.
            Clean datasets, query complex schemas, and generate multi-chart analytical reports in seconds.
          </p>
        </Reveal>

        <Reveal delay={240}>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/dashboard"
              className="flex items-center gap-2 rounded-xl bg-emerald-500 px-6 py-2.5 text-xs font-semibold text-slate-950 shadow-sm transition-all hover:bg-emerald-400"
            >
              Start Free Analysis
              <ArrowRight className="size-3.5" />
            </Link>
            <a
              href="#workflow"
              className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/70 px-5 py-2.5 text-xs font-medium text-slate-300 transition-colors hover:border-slate-700 hover:bg-slate-800 hover:text-white"
            >
              <Play className="size-3.5 text-slate-400" />
              See How It Works
            </a>
          </div>
        </Reveal>

        <Reveal delay={300}>
          <div className="mt-6 flex flex-wrap items-center justify-center gap-6 text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="size-3.5 text-emerald-400" />
              Sandboxed Python engine
            </span>
            <span className="flex items-center gap-1.5">
              <CheckCircle2 className="size-3.5 text-emerald-400" />
              Zero formula errors
            </span>
            <span className="flex items-center gap-1.5">
              <BarChart3 className="size-3.5 text-emerald-400" />
              Automated multi-chart reports
            </span>
          </div>
        </Reveal>

        {/* Dashboard Preview with subtle 3D tilt */}
        <div className="relative mt-12 sm:mt-16 mx-auto max-w-5xl">
          <Parallax speed={-0.08}>
            <TiltCard>
              <div className="rounded-2xl border border-slate-800/90 bg-slate-900/60 p-2 sm:p-3 shadow-2xl backdrop-blur-sm">
                <div className="overflow-hidden rounded-xl border border-slate-800/80 bg-slate-950">
                  <DashboardMockup />
                </div>
              </div>
            </TiltCard>
          </Parallax>
        </div>
      </div>
    </section>
  )
}

const AGENTS = [
  { name: "Task Understanding", icon: Brain, angle: 0 },
  { name: "Workflow Planning", icon: Compass, angle: 60 },
  { name: "Code Generation", icon: Code2, angle: 120 },
  { name: "Sandbox Execution", icon: Terminal, angle: 180 },
  { name: "Validation & QA", icon: ShieldCheck, angle: 240 },
  { name: "Report Synthesis", icon: BarChart3, angle: 300 },
]

function AgentGraphSection() {
  return (
    <section id="architecture" className="relative py-16 sm:py-24 border-b border-slate-800/60 bg-slate-950">
      <div className="mx-auto max-w-6xl px-5">
        <Reveal>
          <div className="text-center">
            <p className="text-xs font-bold uppercase tracking-wider text-emerald-400">
              Multi-Agent Architecture
            </p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-100 sm:text-3xl">
              Coordinated Agents for Verified Accuracy
            </h2>
            <p className="mx-auto mt-3 max-w-xl text-xs leading-relaxed text-slate-400">
              Instead of relying on single-shot LLM guesses, DataPilot routes your task through
              specialized agent nodes that write, execute, inspect, and synthesize verified code.
            </p>
          </div>
        </Reveal>

        <div className="mt-14 relative mx-auto aspect-square w-full max-w-md">
          {/* Subtle connecting lines */}
          <svg viewBox="0 0 400 400" className="absolute inset-0 h-full w-full pointer-events-none">
            {AGENTS.map((agent, i) => {
              const rad = ((agent.angle - 90) * Math.PI) / 180
              const x2 = 200 + Math.cos(rad) * 145
              const y2 = 200 + Math.sin(rad) * 145
              return (
                <line
                  key={agent.name}
                  x1="200"
                  y1="200"
                  x2={x2}
                  y2={y2}
                  stroke="rgba(52, 211, 153, 0.35)"
                  strokeWidth="1.5"
                  className="animate-dash"
                  style={{ animationDelay: `${i * 0.4}s` }}
                />
              )
            })}
            <circle
              cx="200"
              cy="200"
              r="145"
              stroke="rgba(51, 65, 85, 0.45)"
              strokeWidth="1"
              strokeDasharray="4 6"
              fill="none"
            />
          </svg>

          {/* Central Hub */}
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-center">
            <div className="relative flex size-28 flex-col items-center justify-center rounded-full border border-emerald-500/40 bg-slate-900/90 p-3 shadow-lg backdrop-blur-md">
              <span className="size-2 rounded-full bg-emerald-400 mb-1 animate-ping" />
              <p className="text-xs font-bold text-slate-100">DataPilot</p>
              <p className="text-[10px] text-emerald-400 font-mono">Agent Graph</p>
            </div>
          </div>

          {/* Orbiting Agent Nodes */}
          {AGENTS.map((agent, i) => {
            const rad = ((agent.angle - 90) * Math.PI) / 180
            const leftPct = 50 + Math.cos(rad) * 38
            const topPct = 50 + Math.sin(rad) * 38
            const Icon = agent.icon
            return (
              <div
                key={agent.name}
                className="absolute -translate-x-1/2 -translate-y-1/2"
                style={{
                  left: `${leftPct}%`,
                  top: `${topPct}%`,
                  animation: `dp-float ${6 + (i % 2) * 2}s ease-in-out ${i * 0.3}s infinite`,
                }}
              >
                <div className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/90 px-3 py-1.5 text-[11px] font-medium text-slate-200 shadow-md backdrop-blur-md hover:border-emerald-500/40 transition-colors">
                  <Icon className="size-3.5 text-emerald-400 shrink-0" />
                  <span className="whitespace-nowrap">{agent.name}</span>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}

const WORKFLOW_STEPS = [
  {
    num: "01",
    title: "Upload Dataset",
    desc: "Import CSV, XLSX, or extract structured tables from uploaded PDFs and images.",
    icon: UploadCloud,
  },
  {
    num: "02",
    title: "Natural Query",
    desc: "Ask analytical questions, request multi-column transforms, or launch an Auto Report.",
    icon: MessageSquareText,
  },
  {
    num: "03",
    title: "Sandboxed Sandbox",
    desc: "Python AST parsing with syntax self-correction and memory-bounded execution.",
    icon: Code2,
  },
  {
    num: "04",
    title: "Visual Discovery",
    desc: "Explore interactive Plotly charts, styled tables, and publication-ready PDF exports.",
    icon: BarChart3,
  },
]

function WorkflowSection() {
  return (
    <section id="workflow" className="relative py-16 sm:py-24 border-b border-slate-800/60 bg-slate-900/30">
      <div className="mx-auto max-w-6xl px-5">
        <Reveal>
          <div className="text-center">
            <p className="text-xs font-bold uppercase tracking-wider text-emerald-400">
              Interactive Execution
            </p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-100 sm:text-3xl">
              From Raw Spreadsheets to Executive Presentations
            </h2>
          </div>
        </Reveal>

        <div className="relative mt-14 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {/* Animated Connecting Pulse Line */}
          <div
            aria-hidden="true"
            className="pointer-events-none absolute left-12 right-12 top-[34px] hidden h-[2px] bg-slate-800 lg:block"
          >
            <div className="animate-travel absolute top-0 h-full w-28 bg-gradient-to-r from-transparent via-emerald-400 to-transparent" />
          </div>

          {WORKFLOW_STEPS.map((step, index) => {
            const Icon = step.icon
            return (
              <Reveal key={step.num} delay={index * 80}>
                <div className="relative rounded-2xl border border-slate-800/80 bg-slate-950/80 p-5 transition-all hover:border-slate-700">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800/60">
                    <span className="font-mono text-xs font-bold text-emerald-400">{step.num}</span>
                    <span className="flex size-7 items-center justify-center rounded-lg bg-slate-900 border border-slate-800 text-slate-400">
                      <Icon className="size-3.5" />
                    </span>
                  </div>
                  <h3 className="mt-3 text-sm font-semibold text-slate-100">{step.title}</h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-slate-400">{step.desc}</p>
                </div>
              </Reveal>
            )
          })}
        </div>
      </div>
    </section>
  )
}

const CAPABILITIES = [
  {
    icon: MessageSquareText,
    title: "Conversational Data Analysis",
    description:
      "Ask plain English questions to compute metrics, filter rows, or pivot complex datasets without memorizing formulas.",
  },
  {
    icon: BarChart3,
    title: "Multi-Chart Auto Reports",
    description:
      "Understands dataset distributions and generates 2-3 distinct Plotly charts: category breakdowns, histograms, and trend curves.",
  },
  {
    icon: Terminal,
    title: "Sandboxed Python Execution",
    description:
      "Executes real Pandas, NumPy, and Plotly code in a secure sandboxed environment with automated syntax repair.",
  },
  {
    icon: Wand2,
    title: "Automated Data Cleaning",
    description:
      "Instantly identifies and heals missing values, duplicate entries, inconsistent formatting, and malformed headers.",
  },
  {
    icon: History,
    title: "Granular History & Revert",
    description:
      "Step forward and backward through transformations with immutable dataset snapshots and cell diff tracking.",
  },
  {
    icon: Table2,
    title: "PDF & OCR Table Extraction",
    description:
      "Extract structured tabular data from uploaded PDFs, reports, and screenshots straight into active workspace sheets.",
  },
]

function CapabilitiesSection() {
  return (
    <section id="capabilities" className="py-16 sm:py-24 border-b border-slate-800/60 bg-slate-950">
      <div className="mx-auto max-w-6xl px-5">
        <Reveal>
          <div className="text-center">
            <p className="text-xs font-bold uppercase tracking-wider text-emerald-400">
              Core Capabilities
            </p>
            <h2 className="mt-2 text-2xl font-bold tracking-tight text-slate-100 sm:text-3xl">
              Engineered for Real-World Spreadsheets
            </h2>
          </div>
        </Reveal>

        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {CAPABILITIES.map((cap, i) => {
            const Icon = cap.icon
            return (
              <Reveal key={cap.title} delay={i * 60}>
                <div className="group rounded-2xl border border-slate-800/80 bg-slate-900/40 p-5 transition-all hover:border-slate-700 hover:bg-slate-900/70 h-full">
                  <div className="flex size-9 items-center justify-center rounded-xl border border-slate-800 bg-slate-950 text-emerald-400">
                    <Icon className="size-4.5" />
                  </div>
                  <h3 className="mt-4 text-sm font-semibold text-slate-100">{cap.title}</h3>
                  <p className="mt-2 text-xs leading-relaxed text-slate-400">{cap.description}</p>
                </div>
              </Reveal>
            )
          })}
        </div>
      </div>
    </section>
  )
}

function FinalCTA() {
  return (
    <section className="relative overflow-hidden py-16 sm:py-24 bg-slate-950">
      <ShimmerDotBackground />

      <div className="relative mx-auto max-w-4xl px-5 text-center">
        <Reveal>
          <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-8 sm:p-14 backdrop-blur-md">
            <h2 className="text-2xl font-bold tracking-tight text-slate-100 sm:text-4xl">
              Ready to Upgrade Your Data Workflow?
            </h2>
            <p className="mx-auto mt-3 max-w-lg text-xs leading-relaxed text-slate-400 sm:text-sm">
              Explore pre-loaded datasets or upload your own spreadsheets to experience instant
              AI-powered analysis and multi-chart reporting.
            </p>

            <div className="mt-8 flex flex-wrap justify-center gap-3">
              <Link
                href="/dashboard"
                className="flex items-center gap-2 rounded-xl bg-emerald-500 px-6 py-2.5 text-xs font-semibold text-slate-950 hover:bg-emerald-400 transition-colors shadow-sm"
              >
                Launch DataPilot Workspace
                <ArrowRight className="size-3.5" />
              </Link>
              <Link
                href="/login"
                className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900 px-5 py-2.5 text-xs font-medium text-slate-300 hover:border-slate-700 hover:text-white transition-colors"
              >
                Sign In
              </Link>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="border-t border-slate-800/80 bg-slate-950 py-8 text-xs text-slate-500">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-5 sm:flex-row">
        <div className="flex items-center gap-2">
          <Compass className="size-4 text-emerald-400" />
          <span className="font-semibold text-slate-300">DataPilot</span>
          <span className="text-slate-600">·</span>
          <span>Automated Spreadsheet Intelligence</span>
        </div>
        <p>© {new Date().getFullYear()} DataPilot. All rights reserved.</p>
      </div>
    </footer>
  )
}

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-emerald-500/20 selection:text-emerald-300">
      <ScrollProgressBar />
      <Navbar />
      <main>
        <Hero />
        <AgentGraphSection />
        <WorkflowSection />
        <CapabilitiesSection />
        <FinalCTA />
      </main>
      <Footer />
    </div>
  )
}
