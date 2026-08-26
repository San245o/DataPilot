import type { ReactNode } from "react";
import { DataPilotAnimation } from "./DataPilotAnimation";
import { DataPilotLogo } from "./DataPilotLogo";

interface AuthSplitLayoutProps {
  children: ReactNode;
}

export function AuthSplitLayout({
  children,
}: AuthSplitLayoutProps) {
  return (
    <div className="flex min-h-screen flex-col bg-white lg:flex-row">
      {/* =====================================================
          LEFT — DATAPILOT BRAND PANEL
          ===================================================== */}

      <aside
        className="
          relative
          flex
          min-h-[430px]
          w-full
          overflow-hidden
          bg-[var(--auth-panel)]
          px-6
          py-8
          text-[var(--auth-panel-fg)]
          sm:px-10
          lg:min-h-screen
          lg:w-[46%]
          lg:px-14
          lg:py-12
        "
      >
        {/* Background glow */}
        <div className="auth-panel-glow" />

        {/* Subtle grid */}
        <div
          className="
            pointer-events-none
            absolute
            inset-0
            opacity-[0.07]
            [background-image:linear-gradient(rgba(255,255,255,0.35)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.35)_1px,transparent_1px)]
            [background-size:42px_42px]
          "
        />

        <div className="relative z-10 flex w-full flex-col">
          {/* Logo */}
          <div>
            <DataPilotLogo />
          </div>

          {/* Main branding content */}
          <div className="mt-12 max-w-xl lg:mt-auto">
            <div
              className="
                mb-4
                inline-flex
                items-center
                rounded-full
                border
                border-white/10
                bg-white/[0.05]
                px-3
                py-1.5
                text-xs
                font-medium
                tracking-wide
                text-blue-200
                backdrop-blur-sm
              "
            >
              AI-powered spreadsheet intelligence
            </div>

            <h2
              className="
                text-3xl
                font-semibold
                leading-tight
                tracking-tight
                sm:text-4xl
                lg:text-[2.6rem]
              "
            >
              Your data has answers.
              <br />

              <span className="text-[var(--auth-accent)]">
                DataPilot
              </span>{" "}
              helps you find them.
            </h2>

            <p
              className="
                mt-5
                max-w-lg
                text-sm
                leading-7
                text-[var(--auth-panel-muted)]
                sm:text-base
              "
            >
              Turn complex spreadsheets into actionable insights
              with AI-powered analytics, intelligent visualization,
              and natural-language data exploration.
            </p>
          </div>

          {/* Animated DataPilot visualization */}
          <div
            className="
              mt-8
              w-full
              lg:mt-10
            "
          >
            <DataPilotAnimation
              className="
                h-44
                w-full
                sm:h-52
                lg:h-72
              "
            />
          </div>

          {/* Bottom text */}
          <div className="mt-6 hidden items-center gap-2 text-xs text-white/40 lg:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
            <span>Intelligent. Insightful. Data-driven.</span>
          </div>
        </div>
      </aside>

      {/* =====================================================
          RIGHT — AUTHENTICATION FORM
          ===================================================== */}

      <main
        className="
          flex
          min-h-[calc(100vh-430px)]
          flex-1
          items-center
          justify-center
          bg-white
          px-5
          py-12
          text-slate-900
          sm:px-8
          lg:min-h-screen
          lg:px-12
          lg:py-16
        "
      >
        <div
          className="
            w-full
            max-w-md
            animate-[auth-fade-in_0.5s_ease-out]
          "
        >
          {children}
        </div>
      </main>
    </div>
  );
}

export default AuthSplitLayout;