"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowRight,
  BarChart3,
  FileSpreadsheet,
  History,
  Lightbulb,
  Menu,
  MessageSquareText,
  Play,
  Sparkles,
  Table2,
  Upload,
  X,
} from "lucide-react";
import { AppSidebar, Logo } from "@/components/home/app-sidebar";
import { getClientSession } from "@/lib/auth/client";
import "./home.css";

/* ---------------------------------- data --------------------------------- */

const NEW_CARDS = [
  {
    title: "New Analysis",
    description:
      "Upload an Excel or CSV file and let DataPilot turn your data into insights.",
    action: "Start analysis",
    icon: FileSpreadsheet,
    to: "/dashboard",
    featured: true,
  },
  {
    title: "Ask DataPilot",
    description: "Explore your spreadsheet using natural-language questions.",
    action: "Ask a question",
    icon: Sparkles,
    to: "/dashboard",
    featured: false,
  },
  {
    title: "Sample Analysis",
    description: "Explore DataPilot using a sample dataset.",
    action: "Try sample",
    icon: BarChart3,
    to: "/dashboard",
    featured: false,
  },
];

const RECENT_ANALYSES = [
  { name: "Sales Performance", file: "sales_2026.xlsx", when: "Today" },
  { name: "Student Performance", file: "student_results.xlsx", when: "Yesterday" },
  { name: "Product Price Analysis", file: "product_prices.csv", when: "3 days ago" },
];

const TUTORIALS = [
  {
    title: "Getting started with DataPilot",
    description: "Learn how to upload your first spreadsheet and begin analyzing.",
    duration: "3:24",
    icon: Table2,
  },
  {
    title: "Ask questions in natural language",
    description: "See how DataPilot turns simple questions into data analysis.",
    duration: "4:10",
    icon: MessageSquareText,
  },
  {
    title: "From data to visualizations",
    description: "Learn how DataPilot creates meaningful charts from your data.",
    duration: "5:02",
    icon: BarChart3,
  },
  {
    title: "Understanding AI insights",
    description: "Learn how to interpret and use DataPilot's generated insights.",
    duration: "4:45",
    icon: Lightbulb,
  },
];

const QUICK_ACTIONS = [
  { label: "Upload spreadsheet", icon: Upload, to: "/dashboard" },
  { label: "View history", icon: History, to: "/history" },
  { label: "Ask DataPilot", icon: Sparkles, to: "/dashboard" },
];

/* --------------------------------- pieces -------------------------------- */

function SectionHeading({
  title,
  subtitle,
}: {
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="mb-5">
      <h2 className="text-lg font-semibold tracking-tight text-foreground">
        {title}
      </h2>
      {subtitle && (
        <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
      )}
    </div>
  );
}

function NewCard({
  card,
}: {
  card: (typeof NEW_CARDS)[number];
}) {
  const Icon = card.icon;
  return (
    <Link
      href={card.to}
      className={`card-lift group flex flex-col rounded-2xl border p-6 ${
        card.featured
          ? "border-primary/25 bg-card shadow-[0_0_40px_-12px_oklch(0.72_0.17_160/25%)]"
          : "border-border bg-card"
      }`}
    >
      <span className="icon-pop flex size-11 items-center justify-center rounded-xl bg-emerald-soft text-primary ring-1 ring-primary/20">
        <Icon className="size-5" strokeWidth={2} />
      </span>
      <h3 className="mt-4 text-base font-semibold text-foreground">
        {card.title}
      </h3>
      <p className="mt-1.5 flex-1 text-sm leading-relaxed text-muted-foreground">
        {card.description}
      </p>
      <span
        className={`mt-5 inline-flex w-fit items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-all ${
          card.featured
            ? "bg-primary text-primary-foreground group-hover:shadow-[0_0_20px_oklch(0.72_0.17_160/40%)]"
            : "border border-border bg-secondary text-secondary-foreground group-hover:border-primary/40 group-hover:text-primary"
        }`}
      >
        {card.action}
        <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" />
      </span>
    </Link>
  );
}

function RecentList() {
  const hasAnalyses = RECENT_ANALYSES.length > 0;

  return (
    <div className="rounded-2xl border border-border bg-card shadow-card">
      {hasAnalyses ? (
        <>
          <ul className="divide-y divide-border">
            {RECENT_ANALYSES.map((item) => (
              <li key={item.name}>
                <Link
                  href="/dashboard"
                  className="group flex items-center gap-4 px-5 py-4 transition-colors hover:bg-surface-hover"
                >
                  <span className="icon-pop flex size-10 shrink-0 items-center justify-center rounded-lg bg-emerald-soft text-primary ring-1 ring-primary/15">
                    <FileSpreadsheet className="size-4.5" strokeWidth={2} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium text-foreground">
                      {item.name}
                    </span>
                    <span className="block truncate text-xs text-muted-foreground">
                      {item.file}
                    </span>
                  </span>
                  <span className="hidden shrink-0 text-xs text-muted-foreground sm:block">
                    {item.when}
                  </span>
                  <span className="inline-flex shrink-0 items-center gap-1 rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-secondary-foreground transition-colors group-hover:border-primary/40 group-hover:text-primary">
                    Open
                    <ArrowRight className="size-3" />
                  </span>
                </Link>
              </li>
            ))}
          </ul>
          <div className="border-t border-border px-5 py-3.5">
            <Link
              href="/history"
              className="inline-flex items-center gap-1.5 text-sm font-medium text-primary transition-colors hover:text-primary/80"
            >
              View all history
              <ArrowRight className="size-3.5" />
            </Link>
          </div>
        </>
      ) : (
        <div className="flex flex-col items-center px-6 py-14 text-center">
          <span className="flex size-12 items-center justify-center rounded-2xl bg-emerald-soft text-primary ring-1 ring-primary/20">
            <FileSpreadsheet className="size-6" />
          </span>
          <h3 className="mt-4 text-base font-semibold text-foreground">
            No analyses yet
          </h3>
          <p className="mt-1.5 max-w-sm text-sm text-muted-foreground">
            Upload your first spreadsheet and let DataPilot uncover the
            insights.
          </p>
          <Link
            href="/dashboard"
            className="mt-6 inline-flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-shadow hover:shadow-[0_0_20px_oklch(0.72_0.17_160/40%)]"
          >
            Start your first analysis
            <ArrowRight className="size-4" />
          </Link>
        </div>
      )}
    </div>
  );
}

function TutorialCard({
  tutorial,
}: {
  tutorial: (typeof TUTORIALS)[number];
}) {
  const Icon = tutorial.icon;
  return (
    <Link
      href="/tutorials"
      className="card-lift group overflow-hidden rounded-2xl border border-border bg-card"
    >
      {/* Thumbnail area */}
      <div className="relative flex h-32 items-center justify-center overflow-hidden border-b border-border bg-surface-raised">
        <div className="bg-grid-texture absolute inset-0" />
        <div className="absolute inset-0 bg-gradient-to-t from-card/60 to-transparent" />
        <Icon className="relative size-10 text-primary/60 transition-transform duration-300 group-hover:scale-110" strokeWidth={1.5} />
        <span className="absolute bottom-3 left-3 flex size-9 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-[0_0_16px_oklch(0.72_0.17_160/45%)] transition-transform duration-300 group-hover:scale-110">
          <Play className="ml-0.5 size-4" fill="currentColor" />
        </span>
        <span className="absolute bottom-3 right-3 rounded-md bg-background/80 px-1.5 py-0.5 text-[11px] font-medium text-muted-foreground backdrop-blur-sm">
          {tutorial.duration}
        </span>
      </div>
      <div className="p-4">
        <h3 className="text-sm font-semibold leading-snug text-foreground group-hover:text-primary">
          {tutorial.title}
        </h3>
        <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
          {tutorial.description}
        </p>
      </div>
    </Link>
  );
}

/* ---------------------------------- main view --------------------------------- */

export function HomeView() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [firstName, setFirstName] = useState("there");

  useEffect(() => {
    getClientSession().then((session) => {
      if (session) {
        const name = session.name || session.email?.split("@")[0] || "there";
        setFirstName(name.split(" ")[0]);
      }
    });
  }, []);

  return (
    <div className="datapilot-home flex min-h-screen">
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-64 border-r border-border lg:block">
        <AppSidebar />
      </aside>

      {/* Mobile sidebar overlay */}
      {mobileNavOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-background/70 backdrop-blur-sm"
            onClick={() => setMobileNavOpen(false)}
          />
          <aside className="absolute inset-y-0 left-0 w-72 border-r border-border shadow-2xl">
            <AppSidebar onNavigate={() => setMobileNavOpen(false)} />
          </aside>
        </div>
      )}

      {/* Main content */}
      <div className="relative flex-1 lg:pl-64">
        <div className="bg-grid-texture pointer-events-none absolute inset-x-0 top-0 h-96 [mask-image:linear-gradient(to_bottom,black,transparent)]" />
        <div className="pointer-events-none absolute left-1/2 top-0 h-72 w-[36rem] -translate-x-1/2 rounded-full bg-emerald-glow blur-[120px]" />

        {/* Mobile top bar */}
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-border bg-background/85 px-4 py-3 backdrop-blur-md lg:hidden">
          <button
            onClick={() => setMobileNavOpen((v) => !v)}
            className="flex size-9 items-center justify-center rounded-lg border border-border text-secondary-foreground transition-colors hover:bg-secondary"
            aria-label="Toggle navigation"
          >
            {mobileNavOpen ? <X className="size-4" /> : <Menu className="size-4" />}
          </button>
          <Logo size="sm" />
        </header>

        <main className="relative mx-auto w-full max-w-6xl px-5 pb-20 pt-10 sm:px-8 lg:pt-14">
          {/* Greeting */}
          <div>
            <h1 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
              Welcome back{" "}
              <span role="img" aria-label="waving hand">
                👋
              </span>
            </h1>
            <p className="mt-3 text-lg text-foreground/90">
              What would you like to work on today?
            </p>
            <p className="mt-1.5 text-sm text-muted-foreground">
              Start a new analysis, continue where you left off, or learn how to
              get more from your data.
            </p>
          </div>

          {/* New */}
          <section className="mt-12">
            <SectionHeading title="New" />
            <div className="grid gap-4 md:grid-cols-3">
              {NEW_CARDS.map((card) => (
                <NewCard key={card.title} card={card} />
              ))}
            </div>
          </section>

          {/* Recent analyses */}
          <section className="mt-12">
            <SectionHeading
              title="Recent analyses"
              subtitle="Pick up right where you left off."
            />
            <RecentList />
          </section>

          {/* Learn DataPilot */}
          <section className="mt-12">
            <SectionHeading
              title="Learn DataPilot"
              subtitle="Get more from your spreadsheets with DataPilot."
            />
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {TUTORIALS.map((tutorial) => (
                <TutorialCard key={tutorial.title} tutorial={tutorial} />
              ))}
            </div>
          </section>

          {/* Quick actions */}
          <section className="mt-12">
            <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-border bg-card px-5 py-4">
              <span className="text-sm font-medium text-muted-foreground">
                Quick actions
              </span>
              <div className="flex flex-wrap gap-2">
                {QUICK_ACTIONS.map(({ label, icon: Icon, to }) => (
                  <Link
                    key={label}
                    href={to}
                    className="group inline-flex items-center gap-2 rounded-lg border border-border bg-secondary px-3.5 py-2 text-sm text-secondary-foreground transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:text-primary"
                  >
                    <Icon className="size-4 text-primary/80" />
                    {label}
                  </Link>
                ))}
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
