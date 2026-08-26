import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  Sparkles,
  Play,
  ArrowRight,
  Menu,
  X,
  UploadCloud,
  MessageSquareText,
  Brain,
  Terminal,
  BarChart3,
  Wand2,
  FileScan,
  LineChart,
  History,
  Lock,
  Compass,
} from "lucide-react";
import {
  Parallax,
  Reveal,
  TiltCard,
  useScrollProgress,
} from "@/components/parallax";
import {
  GridBackdrop,
  FloatingIcons,
  Particles,
  Atmosphere,
  DataLayer,
} from "@/components/decor";
import { DashboardMockup } from "@/components/dashboard-mockup";
import poster from "@/assets/demo-poster.jpg";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      {
        title: "DataPilot — Multi-Agent AI for Data Analysis & Visualization",
      },
      {
        name: "description",
        content:
          "DataPilot uses multi-agent AI to clean, analyze, transform and visualize your data through simple natural language commands.",
      },
      {
        property: "og:title",
        content: "DataPilot — Turn Raw Data Into Meaningful Insights",
      },
      {
        property: "og:description",
        content:
          "Upload a dataset, ask in plain language, and let multi-agent AI clean, analyze and visualize it.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Landing,
});

const NAV = [
  { label: "Home", href: "#home" },
  { label: "Features", href: "#features" },
  { label: "How It Works", href: "#how" },
  { label: "Technology", href: "#technology" },
  { label: "Demo", href: "#demo" },
];

function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <span className="relative grid size-9 place-items-center rounded-xl border border-primary/40 bg-primary/15">
        <Compass className="size-5 text-accent" strokeWidth={1.8} />
        <span className="absolute inset-0 rounded-xl bg-accent/10 blur-md" />
      </span>
      <span className="text-[17px] font-semibold tracking-tight">
        DataPilot
      </span>
    </span>
  );
}

function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });

    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 transition-all duration-300 ${
        scrolled
          ? "glass border-b shadow-lg"
          : "border-b border-transparent"
      }`}
      style={scrolled ? undefined : { background: "transparent" }}
    >
      <nav className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5">
        <a href="#home" aria-label="DataPilot home">
          <Logo />
        </a>

        <ul className="hidden items-center gap-7 lg:flex">
          {NAV.map((n) => (
            <li key={n.label}>
              <a
                href={n.href}
                className="text-sm text-muted-foreground transition-colors hover:text-foreground"
              >
                {n.label}
              </a>
            </li>
          ))}
        </ul>

        {/* Desktop Login + Signup */}
        <div className="hidden items-center gap-2 sm:flex">
          <Link
            to="/login"
            className="rounded-lg px-4 py-2 text-sm font-medium text-foreground/85 transition-colors hover:bg-surface-2/60 hover:text-foreground"
          >
            Log In
          </Link>

          <Link
            to="/signup"
            className="rounded-lg bg-gradient-to-r from-primary to-accent px-4 py-2 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/25 transition-transform hover:-translate-y-0.5 active:translate-y-0"
          >
            Sign Up
          </Link>
        </div>

        {/* Mobile Menu Button */}
        <button
          className="lg:hidden"
          onClick={() => setOpen((o) => !o)}
          aria-label="Toggle navigation"
        >
          {open ? (
            <X className="size-5" />
          ) : (
            <Menu className="size-5" />
          )}
        </button>
      </nav>

      {/* Mobile Navigation */}
      {open && (
        <div className="glass border-t px-5 pb-5 pt-2 lg:hidden">
          <ul className="space-y-1">
            {NAV.map((n) => (
              <li key={n.label}>
                <a
                  href={n.href}
                  onClick={() => setOpen(false)}
                  className="block rounded-lg px-2 py-2.5 text-sm text-muted-foreground hover:bg-surface-2/50 hover:text-foreground"
                >
                  {n.label}
                </a>
              </li>
            ))}
          </ul>

          {/* Mobile Login + Signup */}
          <div className="mt-3 grid grid-cols-2 gap-2">
            <Link
              to="/login"
              onClick={() => setOpen(false)}
              className="rounded-lg border border-border px-4 py-2 text-center text-sm transition-colors hover:bg-surface-2/60"
            >
              Log In
            </Link>

            <Link
              to="/signup"
              onClick={() => setOpen(false)}
              className="rounded-lg bg-gradient-to-r from-primary to-accent px-4 py-2 text-center text-sm font-semibold text-primary-foreground"
            >
              Sign Up
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}

function SectionHeading({
  eyebrow,
  title,
  sub,
}: {
  eyebrow?: string;
  title: string;
  sub?: string;
}) {
  return (
    <Reveal className="mx-auto max-w-2xl text-center">
      {eyebrow && (
        <p className="mb-3 text-xs font-medium uppercase tracking-[0.18em] text-accent/80">
          {eyebrow}
        </p>
      )}

      <h2 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
        {title}
      </h2>

      {sub && (
        <p className="mt-3 text-pretty text-[15px] text-muted-foreground">
          {sub}
        </p>
      )}
    </Reveal>
  );
}

function Hero() {
  return (
    <section
      id="home"
      className="relative overflow-hidden pb-12 pt-28 sm:pb-16 sm:pt-32"
    >
      <GridBackdrop />
      <Atmosphere />
      <DataLayer />
      <FloatingIcons />
      <Particles count={14} />

      <div className="relative mx-auto max-w-6xl px-5">
        <Parallax speed={-0.08} className="text-center">
          <span className="glass inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-xs font-medium text-foreground/85">
            <Sparkles className="size-3.5 text-accent" />
            AI-Powered Data Analytics
          </span>

          <h1 className="mx-auto mt-6 max-w-4xl text-balance text-4xl font-semibold leading-[1.08] tracking-tight sm:text-6xl lg:text-[68px]">
            Turn Raw Data Into
            <br />
            <span className="text-gradient">
              Meaningful Insights.
            </span>
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-pretty text-base leading-relaxed text-muted-foreground sm:text-lg">
            DataPilot uses multi-agent AI to clean, analyze, transform, and
            visualize your data through simple natural language commands.
          </p>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <a
              href="#demo"
              className="group inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-primary to-accent px-6 py-3 text-sm font-semibold text-primary-foreground shadow-xl shadow-primary/30 transition-all hover:-translate-y-0.5 hover:shadow-2xl active:translate-y-0"
            >
              Try DataPilot
              <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
            </a>

            <a
              href="#demo"
              className="glass inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold text-foreground transition-all hover:-translate-y-0.5 hover:border-accent/40"
            >
              <Play className="size-4 text-accent" />
              Watch Demo
            </a>
          </div>

          <p className="mt-6 font-mono text-[12px] tracking-wide text-muted-foreground/80">
            Built with Next.js • FastAPI • LangGraph • Pandas • Plotly
          </p>
        </Parallax>

        <Parallax
          speed={-0.16}
          scaleWith={-0.03}
          className="relative mt-14"
        >
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-6 -top-6 bottom-0 rounded-[2rem] blur-3xl"
            style={{
              background:
                "radial-gradient(60% 60% at 50% 40%, color-mix(in oklab, var(--blue) 35%, transparent), transparent 70%)",
            }}
          />

          <TiltCard className="relative">
            <div className="glow-soft rounded-2xl">
              <DashboardMockup />
            </div>
          </TiltCard>
        </Parallax>
      </div>
    </section>
  );
}

function DemoSection() {
  return (
    <section
      id="demo"
      className="relative overflow-hidden bg-surface/20 py-14 sm:py-16"
    >
      <Parallax
        speed={0.35}
        className="pointer-events-none absolute inset-0"
      >
        <div className="grid-pattern absolute inset-[-20%] opacity-25 [mask-image:radial-gradient(60%_60%_at_50%_50%,black,transparent)]" />
      </Parallax>

      <div className="relative mx-auto max-w-5xl px-5">
        <SectionHeading
          eyebrow="Demo"
          title="See DataPilot In Action"
          sub="From raw spreadsheets to actionable insights — with a simple natural-language request."
        />

        <Reveal className="mt-8">
          <Parallax speed={-0.06}>
            <div className="glow-ring group relative overflow-hidden rounded-2xl border border-primary/25">
              <video
                className="aspect-video w-full bg-deep object-cover"
                poster={poster}
                controls
                preload="none"
              >
                <source
                  src="/datapilot-demo.mp4"
                  type="video/mp4"
                />
              </video>

              <div className="pointer-events-none absolute inset-0 grid place-items-center transition-opacity group-hover:opacity-0">
                <span className="grid size-16 place-items-center rounded-full border border-accent/40 bg-background/60 backdrop-blur-md">
                  <Play className="size-6 translate-x-0.5 text-accent" />
                </span>
              </div>
            </div>
          </Parallax>

          <p className="mt-4 text-center font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground/70">
            Upload → Ask → AI agents process → Clean → Visualize → Insights
          </p>
        </Reveal>
      </div>
    </section>
  );
}

const STEPS = [
  { t: "Upload", icon: UploadCloud },
  { t: "Ask", icon: MessageSquareText },
  { t: "Understand", icon: Brain },
  { t: "Execute", icon: Terminal },
  { t: "Visualize", icon: BarChart3 },
  { t: "Insights", icon: Sparkles },
];

const AGENTS = [
  "Task Understanding",
  "Planning",
  "Code Generation",
  "Execution",
  "Validation",
  "Result Explanation",
];

function AgentGraph() {
  return (
    <div className="relative mx-auto aspect-square w-full max-w-md">
      <svg
        viewBox="0 0 400 400"
        className="absolute inset-0 h-full w-full"
      >
        {AGENTS.map((_, i) => {
          const a =
            (i / AGENTS.length) * Math.PI * 2 - Math.PI / 2;

          return (
            <line
              key={i}
              x1="200"
              y1="200"
              x2={200 + Math.cos(a) * 150}
              y2={200 + Math.sin(a) * 150}
              stroke="var(--blue)"
              strokeOpacity="0.5"
              strokeWidth="1.4"
              className="animate-dash"
              style={{
                animationDelay: `${i * 0.4}s`,
              }}
            />
          );
        })}

        <circle
          cx="200"
          cy="200"
          r="150"
          stroke="var(--blue)"
          strokeOpacity="0.15"
          fill="none"
        />
      </svg>

      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
        <span
          aria-hidden
          className="animate-ring absolute inset-[-14px] rounded-full border border-accent/40"
        />

        <div className="glow-ring relative grid size-28 place-items-center rounded-full border border-accent/40 bg-surface/70 text-center text-sm font-semibold backdrop-blur">
          DataPilot
          <br />
          AI
        </div>
      </div>

      {AGENTS.map((label, i) => {
        const a =
          (i / AGENTS.length) * Math.PI * 2 - Math.PI / 2;

        return (
          <div
            key={label}
            className="absolute -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded-full border border-primary/25 bg-surface/70 px-3 py-1.5 text-[11px] font-medium text-foreground/90 backdrop-blur"
            style={{
              left: `${50 + Math.cos(a) * 39}%`,
              top: `${50 + Math.sin(a) * 39}%`,
              animation: `dp-float ${7 + (i % 3)}s ease-in-out ${i * 0.5}s infinite`,
            }}
          >
            {label}
          </div>
        );
      })}
    </div>
  );
}

function HowItWorks() {
  return (
    <section
      id="how"
      className="relative overflow-hidden py-14 sm:py-16"
    >
      <Parallax
        speed={0.3}
        className="pointer-events-none absolute inset-0"
      >
        <div
          className="absolute inset-0 opacity-70"
          style={{
            background:
              "radial-gradient(50% 40% at 20% 20%, color-mix(in oklab, var(--blue) 18%, transparent), transparent 70%)",
          }}
        />
      </Parallax>

      <div className="relative mx-auto max-w-6xl px-5">
        <SectionHeading
          eyebrow="How it works"
          title="From Question to Insight in Seconds."
        />

        <div className="mt-10 grid items-center gap-10 lg:grid-cols-[1.05fr_1fr]">
          <Reveal>
            <div className="glass-card relative rounded-2xl p-5 sm:p-6">
              <svg
                aria-hidden
                viewBox="0 0 100 2"
                preserveAspectRatio="none"
                className="pointer-events-none absolute left-6 right-6 top-[54px] hidden h-0.5 text-accent/50 sm:block"
              >
                <line
                  x1="0"
                  y1="1"
                  x2="100"
                  y2="1"
                  stroke="currentColor"
                  strokeWidth="2"
                  className="animate-dash"
                />
              </svg>

              <div
                aria-hidden
                className="pointer-events-none absolute left-6 right-6 top-[54px] hidden h-0.5 sm:block"
              >
                {[0, 1, 2].map((k) => (
                  <span
                    key={k}
                    className="animate-travel absolute top-1/2 size-1.5 -translate-y-1/2 rounded-full bg-accent"
                    style={{
                      animationDelay: `${k * 2}s`,
                      boxShadow:
                        "0 0 10px color-mix(in oklab, var(--blue) 90%, transparent)",
                    }}
                  />
                ))}
              </div>

              <div className="relative grid grid-cols-3 gap-4 sm:grid-cols-6">
                {STEPS.map((s, i) => (
                  <div
                    key={s.t}
                    className="flex flex-col items-center gap-2 text-center"
                  >
                    <span
                      className="grid size-10 place-items-center rounded-xl border border-primary/25 bg-surface/80 backdrop-blur"
                      style={{
                        animation: `dp-pulse-soft 3.6s ease-in-out ${i * 0.3}s infinite`,
                      }}
                    >
                      <s.icon
                        className="size-4 text-accent"
                        strokeWidth={1.7}
                      />
                    </span>

                    <span className="text-[11px] font-medium text-foreground/85">
                      {s.t}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <p className="mt-5 max-w-md text-[14px] leading-relaxed text-muted-foreground">
              Specialized agents handle understanding, planning, code
              generation, execution, validation and explanation inside one
              coordinated graph.
            </p>
          </Reveal>

          <Reveal delay={100}>
            <Parallax speed={-0.1}>
              <AgentGraph />
            </Parallax>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

const FEATURES = [
  {
    icon: MessageSquareText,
    t: "Natural Language Analysis",
    d: "Ask questions in plain language instead of writing code.",
  },
  {
    icon: Wand2,
    t: "Automated Data Cleaning",
    d: "Missing values, duplicates and bad types fixed automatically.",
  },
  {
    icon: FileScan,
    t: "PDF & Image Table Extraction",
    d: "Pull clean tables out of PDFs and scans with OCR.",
  },
  {
    icon: LineChart,
    t: "Interactive Visualizations",
    d: "Plotly charts and dashboards generated on demand.",
  },
  {
    icon: Lock,
    t: "Secure Code Execution",
    d: "Generated Python runs inside a sandboxed environment.",
  },
  {
    icon: History,
    t: "Version History & Rollback",
    d: "Compare transformations and revert any change instantly.",
  },
];

function Features() {
  return (
    <section
      id="features"
      className="relative overflow-hidden bg-surface/15 py-14 sm:py-16"
    >
      <FloatingIcons
        specs={[
          {
            x: "7%",
            y: "12%",
            size: 20,
            speed: 0.7,
            icon: 2,
          },
          {
            x: "89%",
            y: "40%",
            size: 26,
            speed: 0.5,
            icon: 0,
            hideSm: true,
          },
        ]}
      />

      <div className="relative mx-auto max-w-6xl px-5">
        <SectionHeading
          eyebrow="Features"
          title="Everything You Need to Work With Data"
        />

        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f, i) => (
            <Reveal key={f.t} delay={i * 60}>
              <div className="glass-card hover-glow group h-full rounded-2xl p-5">
                <span className="grid size-10 place-items-center rounded-xl border border-primary/25 bg-primary/10 transition-colors group-hover:bg-accent/15">
                  <f.icon
                    className="size-[18px] text-accent"
                    strokeWidth={1.7}
                  />
                </span>

                <h3 className="mt-4 text-[15px] font-semibold">
                  {f.t}
                </h3>

                <p className="mt-1.5 text-[13px] leading-relaxed text-muted-foreground">
                  {f.d}
                </p>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

const TECH = [
  "Next.js",
  "FastAPI",
  "LangGraph / AutoGen",
  "Google Gemini AI",
  "Pandas",
  "NumPy",
  "Plotly",
  "Matplotlib / Pyplot",
  "OCR / Azure Computer Vision",
];

const ARCH = [
  "User",
  "Next.js",
  "FastAPI",
  "Multi-Agent AI",
  "Data Processing",
  "Visualization",
  "Results",
];

function Technology() {
  return (
    <section
      id="technology"
      className="relative overflow-hidden py-14 sm:py-16"
    >
      <Parallax
        speed={0.32}
        className="pointer-events-none absolute inset-0"
      >
        <div className="grid-pattern absolute inset-[-20%] opacity-30 [mask-image:radial-gradient(60%_60%_at_50%_40%,black,transparent)]" />
      </Parallax>

      <Particles count={10} />

      <div className="relative mx-auto max-w-5xl px-5">
        <SectionHeading
          eyebrow="Technology"
          title="Powered by a Modern AI Stack"
        />

        <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-3">
          {TECH.map((t, i) => (
            <Reveal key={t} delay={i * 40}>
              <div className="glass-card hover-glow flex h-full items-center gap-3 rounded-xl px-4 py-3 text-[13px] font-medium">
                <span className="size-1.5 rounded-full bg-accent shadow-[0_0_10px_var(--cyan)]" />
                {t}
              </div>
            </Reveal>
          ))}
        </div>

        <Reveal className="mt-8">
          <div className="glass-card rounded-2xl p-5 sm:p-7">
            <div className="flex flex-col items-stretch gap-2 md:flex-row md:items-center">
              {ARCH.map((a, i) => (
                <div
                  key={a}
                  className="flex flex-1 flex-col items-center gap-2 md:flex-row"
                >
                  <div
                    className={`w-full rounded-xl border px-3 py-3 text-center text-[12px] font-medium ${
                      i === 3
                        ? "border-accent/45 bg-accent/10 text-accent"
                        : "border-border bg-surface-2/35 text-foreground/85"
                    }`}
                  >
                    {a}
                  </div>

                  {i < ARCH.length - 1 && (
                    <svg
                      viewBox="0 0 40 12"
                      className="h-3 w-6 shrink-0 rotate-90 text-accent/60 md:rotate-0"
                    >
                      <line
                        x1="0"
                        y1="6"
                        x2="40"
                        y2="6"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        className="animate-dash"
                      />
                    </svg>
                  )}
                </div>
              ))}
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

function FinalCTA() {
  return (
    <section className="relative overflow-hidden py-16">
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: "var(--gradient-hero)",
          opacity: 0.85,
        }}
      />

      <FloatingIcons
        specs={[
          {
            x: "8%",
            y: "26%",
            size: 24,
            speed: 0.65,
            icon: 0,
          },
          {
            x: "80%",
            y: "70%",
            size: 26,
            speed: 0.5,
            icon: 5,
            hideSm: true,
          },
        ]}
      />

      <Particles count={12} />

      <div className="relative mx-auto max-w-3xl px-5 text-center">
        <Reveal>
          <h2 className="text-balance text-3xl font-semibold tracking-tight sm:text-5xl">
            Your Data. Your Questions.{" "}
            <span className="text-gradient">DataPilot.</span>
          </h2>

          <p className="mx-auto mt-4 max-w-xl text-pretty text-[15px] text-muted-foreground">
            Make data analysis simpler with natural-language interaction and
            intelligent automation.
          </p>

          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <a
              href="#home"
              className="group inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-primary to-accent px-6 py-3 text-sm font-semibold text-primary-foreground shadow-xl shadow-primary/30 transition-all hover:-translate-y-0.5 active:translate-y-0"
            >
              Explore DataPilot
              <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
            </a>

            <a
              href="#demo"
              className="glass inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold transition-all hover:-translate-y-0.5 hover:border-accent/40"
            >
              <Play className="size-4 text-accent" />
              Watch Demo
            </a>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-border/60 bg-deep/40">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-5 py-10 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <Logo />

          <p className="mt-2 text-[13px] text-muted-foreground">
            AI-powered data analysis and visualization.
          </p>
        </div>

        <ul className="flex flex-wrap gap-x-6 gap-y-2">
          {NAV.map((n) => (
            <li key={n.label}>
              <a
                href={n.href}
                className="text-[13px] text-muted-foreground transition-colors hover:text-foreground"
              >
                {n.label}
              </a>
            </li>
          ))}
        </ul>
      </div>

      <div className="border-t border-border/50 px-5 py-4 text-center text-[12px] text-muted-foreground/70">
        © {new Date().getFullYear()} DataPilot
      </div>
    </footer>
  );
}

function ScrollProgress() {
  const p = useScrollProgress();

  return (
    <div className="fixed inset-x-0 top-0 z-[60] h-0.5 bg-transparent">
      <div
        className="h-full bg-gradient-to-r from-primary to-accent transition-[width] duration-150"
        style={{ width: `${p * 100}%` }}
      />
    </div>
  );
}

function Landing() {
  return (
    <div className="relative min-h-screen bg-background">
      <ScrollProgress />

      <Navbar />

      <main>
        <Hero />
        <DemoSection />
        <HowItWorks />
        <Features />
        <Technology />
        <FinalCTA />
      </main>

      <Footer />
    </div>
  );
}