"use client";

/**
 * DataPilotAnimation
 * Lightweight SVG/CSS animation for the auth split panel.
 */
export function DataPilotAnimation({ className = "" }: { className?: string }) {
  const cells = Array.from({ length: 24 });
  const bars = [34, 52, 28, 68, 44, 82];

  return (
    <div className={`pointer-events-none select-none ${className}`} aria-hidden="true">
      <svg viewBox="0 0 420 300" className="h-full w-full overflow-visible">
        {/* Spreadsheet grid */}
        <g transform="translate(8,26)">
          <rect
            x="0"
            y="-18"
            width="168"
            height="16"
            rx="4"
            className="fill-[var(--auth-panel-line)]"
          />
          {cells.map((_, i) => {
            const col = i % 4;
            const row = Math.floor(i / 4);
            return (
              <rect
                key={i}
                x={col * 42}
                y={row * 26}
                width="38"
                height="22"
                rx="4"
                className="fill-[var(--auth-cell)] stroke-[var(--auth-panel-line)] dp-cell"
                style={{ animationDelay: `${i * 90}ms` }}
              />
            );
          })}
        </g>

        {/* Flow path from sheet to AI core */}
        <path
          id="dp-flow"
          d="M180 130 C 218 130, 218 118, 246 118"
          className="stroke-[var(--auth-accent-soft)] dp-path"
          fill="none"
          strokeWidth="1.5"
        />
        {[0, 1, 2].map((i) => (
          <circle key={i} r="3.5" className="fill-[var(--auth-accent)] dp-dot">
            <animateMotion
              dur="2.6s"
              begin={`${i * 0.85}s`}
              repeatCount="indefinite"
              path="M180 130 C 218 130, 218 118, 246 118"
            />
          </circle>
        ))}

        {/* AI core */}
        <g transform="translate(272,118)">
          <circle r="26" className="fill-none stroke-[var(--auth-accent-soft)] dp-pulse" />
          <circle r="16" className="fill-[var(--auth-accent)]/15 stroke-[var(--auth-accent)]" />
          <g className="dp-spin">
            {[0, 60, 120, 180, 240, 300].map((a) => (
              <circle
                key={a}
                cx={Math.cos((a * Math.PI) / 180) * 26}
                cy={Math.sin((a * Math.PI) / 180) * 26}
                r="2.5"
                className="fill-[var(--auth-accent)]"
              />
            ))}
          </g>
          <text
            textAnchor="middle"
            y="4"
            className="fill-[var(--auth-panel-fg)] text-[9px] font-semibold tracking-widest"
          >
            AI
          </text>
        </g>

        {/* Chart forming */}
        <g transform="translate(312,232)">
          <line x1="-6" y1="4" x2="108" y2="4" className="stroke-[var(--auth-panel-line)]" />
          {bars.map((h, i) => (
            <rect
              key={i}
              x={i * 18}
              y={-h}
              width="11"
              height={h}
              rx="3"
              className="fill-[var(--auth-accent)] dp-bar"
              style={{ animationDelay: `${600 + i * 160}ms`, transformOrigin: `0px 4px` }}
            />
          ))}
          <polyline
            points="5,-40 23,-58 41,-34 59,-74 77,-50 95,-88"
            fill="none"
            strokeWidth="2"
            strokeLinecap="round"
            className="stroke-[var(--auth-panel-fg)] dp-line"
          />
        </g>

        {/* Insight badge */}
        <g transform="translate(300,64)" className="dp-insight">
          <rect
            x="0"
            y="0"
            width="112"
            height="30"
            rx="15"
            className="fill-[var(--auth-cell)] stroke-[var(--auth-accent-soft)]"
          />
          <circle cx="18" cy="15" r="5" className="fill-[var(--auth-accent)]" />
          <text x="32" y="19" className="fill-[var(--auth-panel-fg)] text-[10px]">
            +18% revenue
          </text>
        </g>
      </svg>
    </div>
  );
}

export default DataPilotAnimation;
