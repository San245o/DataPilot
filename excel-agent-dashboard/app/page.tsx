"use client"

import { useEffect, useRef } from "react"
import Link from "next/link"
import { 
  Sparkles, 
  TimerOff, 
  AlertCircle, 
  Zap, 
  Brain, 
  Bot, 
  Brush, 
  Activity, 
  Binary, 
  Table2, 
  CloudSync, 
  Database, 
  ShieldCheck, 
  Share2, 
  Globe 
} from "lucide-react"

export default function LandingPage() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const heroRef = useRef<HTMLDivElement | null>(null)

  // WebGL Shader Background Logic
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const gl = (canvas.getContext("webgl") || canvas.getContext("experimental-webgl")) as any
    if (!gl) return

    const vs = `
      attribute vec2 a_position;
      varying vec2 v_texCoord;
      void main() {
        v_texCoord = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `

    const fs = `
      precision highp float;
      uniform float u_time;
      uniform vec2 u_resolution;
      uniform vec2 u_mouse;
      varying vec2 v_texCoord;

      void main() {
          vec2 uv = v_texCoord;
          vec2 mouse = u_mouse / u_resolution;
          
          // Parallax effect based on mouse and time
          vec2 shiftedUv = uv + mouse * 0.02;
          
          // Create a scrolling grid pattern
          vec2 gridUv = shiftedUv * 20.0;
          gridUv.y += u_time * 0.2;
          
          vec2 grid = abs(fract(gridUv - 0.5) - 0.5) / fwidth(gridUv);
          float line = min(grid.x, grid.y);
          float gridPattern = 1.0 - min(line, 1.0);
          
          // Background glow
          float glow = distance(uv, vec2(0.5) + mouse * 0.1);
          vec3 color = mix(vec3(0.02, 0.05, 0.1), vec3(0.06, 0.1, 0.15), 1.0 - glow);
          
          // Add the grid in a subtle way
          color += gridPattern * vec3(0.06, 0.72, 0.5) * 0.15;
          
          // Add "data stream" particles
          float particle = sin(uv.x * 50.0 + u_time * 2.0) * cos(uv.y * 30.0 - u_time * 1.5);
          color += smoothstep(0.98, 1.0, particle) * vec3(0.4, 0.9, 1.0) * 0.3;

          gl_FragColor = vec4(color, 1.0);
      }
    `

    const compileShader = (type: number, src: string) => {
      const shader = gl.createShader(type)
      if (!shader) return null
      gl.shaderSource(shader, src)
      gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(shader))
        gl.deleteShader(shader)
        return null
      }
      return shader
    }

    const vertexShader = compileShader(gl.VERTEX_SHADER, vs)
    const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fs)
    if (!vertexShader || !fragmentShader) return

    const prog = gl.createProgram()
    if (!prog) return
    gl.attachShader(prog, vertexShader)
    gl.attachShader(prog, fragmentShader)
    gl.linkProgram(prog)
    gl.useProgram(prog)

    const buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW
    )

    const pos = gl.getAttribLocation(prog, "a_position")
    gl.enableVertexAttribArray(pos)
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0)

    const uTime = gl.getUniformLocation(prog, "u_time")
    const uRes = gl.getUniformLocation(prog, "u_resolution")
    const uMouse = gl.getUniformLocation(prog, "u_mouse")

    let mouse = { x: canvas.width / 2, y: canvas.height / 2 }

    const handleMouseMove = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      if (rect.width && rect.height) {
        const nx = (event.clientX - rect.left) / rect.width
        const ny = 1.0 - (event.clientY - rect.top) / rect.height
        mouse.x = nx * canvas.width
        mouse.y = ny * canvas.height
      }
    }

    window.addEventListener("mousemove", handleMouseMove)

    const syncSize = () => {
      const w = canvas.clientWidth || 1280
      const h = canvas.clientHeight || 720
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
    }

    syncSize()
    window.addEventListener("resize", syncSize)

    let reqId = 0
    const render = (t: number) => {
      gl.viewport(0, 0, canvas.width, canvas.height)
      if (uTime) gl.uniform1f(uTime, t * 0.001)
      if (uRes) gl.uniform2f(uRes, canvas.width, canvas.height)
      if (uMouse) gl.uniform2f(uMouse, mouse.x, mouse.y)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      reqId = requestAnimationFrame(render)
    }

    reqId = requestAnimationFrame(render)

    return () => {
      cancelAnimationFrame(reqId)
      window.removeEventListener("mousemove", handleMouseMove)
      window.removeEventListener("resize", syncSize)
    }
  }, [])

  // Parallax Scrolling effect
  useEffect(() => {
    const handleScroll = () => {
      const layers = document.querySelectorAll(".parallax-layer")
      const scrolled = window.pageYOffset

      layers.forEach((layer) => {
        const speed = parseFloat(layer.getAttribute("data-speed") || "0.1")
        const offset = scrolled * speed
        ;(layer as HTMLElement).style.transform = `translateY(${offset}px)`
      })
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  // Interactive Card Highlights & Entrance Animation
  useEffect(() => {
    // Card highlights
    const cards = document.querySelectorAll(".glass-card")
    const handleCardMouseMove = (e: Event) => {
      const card = e.currentTarget as HTMLElement
      const rect = card.getBoundingClientRect()
      const mouseEvt = e as MouseEvent
      const x = mouseEvt.clientX - rect.left
      const y = mouseEvt.clientY - rect.top
      
      card.style.setProperty("--mouse-x", `${x}px`)
      card.style.setProperty("--mouse-y", `${y}px`)
    }

    cards.forEach((card) => {
      card.addEventListener("mousemove", handleCardMouseMove)
    })

    // Entrance Animation
    const heroContent = heroRef.current
    if (heroContent) {
      heroContent.style.opacity = "0"
      heroContent.style.transform = "translateY(30px)"
      
      const timer = setTimeout(() => {
        heroContent.style.transition = "all 1.2s cubic-bezier(0.16, 1, 0.3, 1)"
        heroContent.style.opacity = "1"
        heroContent.style.transform = "translateY(0px)"
      }, 100)
      return () => clearTimeout(timer)
    }

    return () => {
      cards.forEach((card) => {
        card.removeEventListener("mousemove", handleCardMouseMove)
      })
    }
  }, [])

  return (
    <div className="font-sans antialiased text-[#dae2fd] bg-[#020617] min-h-screen overflow-x-hidden selection:bg-[#4edea3]/30">
      
      {/* Custom Styles Injector */}
      <style jsx global>{`
        body {
          background-color: #020617;
          color: #dae2fd;
          overflow-x: hidden;
          scroll-behavior: smooth;
        }

        .parallax-layer {
          transition: transform 0.2s cubic-bezier(0, 0, 0.2, 1);
          will-change: transform;
        }

        .glass-card {
          background: rgba(15, 23, 42, 0.6);
          backdrop-filter: blur(12px);
          border: 1px solid rgba(51, 65, 85, 0.5);
          position: relative;
          overflow: hidden;
        }

        .glass-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 1px;
          background: linear-gradient(90deg, transparent, #4edea3, transparent);
        }

        .ai-glow {
          box-shadow: inset 0 0 10px rgba(78, 222, 163, 0.1), 0 0 20px rgba(78, 222, 163, 0.05);
          border: 1px solid #4edea3 !important;
        }

        .data-grid-overlay {
          background-image: radial-gradient(#1e293b 1px, transparent 1px);
          background-size: 24px 24px;
          opacity: 0.2;
        }

        .pulse-dot {
          width: 8px;
          height: 8px;
          background: #4edea3;
          border-radius: 50%;
          display: inline-block;
          animation: pulse-kf 2s infinite;
        }

        @keyframes pulse-kf {
          0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(78, 222, 163, 0.7); }
          70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(78, 222, 163, 0); }
          100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(78, 222, 163, 0); }
        }

        .hover-lift {
          transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .hover-lift:hover {
          transform: translateY(-4px);
          box-shadow: 0 10px 30px -10px rgba(78, 222, 163, 0.3);
        }
      `}</style>

      {/* Global Background Shader */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="w-full h-full opacity-40">
          <canvas ref={canvasRef} className="block w-full h-full"></canvas>
        </div>
        <div className="absolute inset-0 data-grid-overlay"></div>
      </div>

      {/* Navigation */}
      <header className="fixed top-0 w-full z-50 bg-[#0b1326]/10 backdrop-blur-md border-b border-[#3c4a42]/20 shadow-sm">
        <nav className="flex justify-between items-center px-6 h-16 w-full max-w-7xl mx-auto">
          <div className="flex items-center gap-2">
            <span className="font-extrabold text-2xl text-[#4edea3] tracking-tighter">DataPilot</span>
            <div className="pulse-dot"></div>
          </div>
          <div className="hidden md:flex items-center gap-10">
            <a className="text-sm text-[#4edea3] font-bold border-b-2 border-[#4edea3] pb-1" href="#hero">Features</a>
            <a className="text-sm text-[#bbcabf] hover:text-[#4edea3] transition-colors" href="#">Case Studies</a>
            <a className="text-sm text-[#bbcabf] hover:text-[#4edea3] transition-colors" href="#">Documentation</a>
            <a className="text-sm text-[#bbcabf] hover:text-[#4edea3] transition-colors" href="#">Pricing</a>
          </div>
          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="text-xs font-semibold text-[#bbcabf] hover:text-white cursor-pointer active:scale-95 transition-all">Login</Link>
            <Link href="/dashboard" className="bg-[#4edea3] text-[#003824] px-4 py-2 rounded-lg text-xs font-bold cursor-pointer active:scale-95 transition-all hover:brightness-110 shadow-[0_0_15px_rgba(78,222,163,0.3)]">Launch App</Link>
          </div>
        </nav>
      </header>

      <main className="relative z-10 pt-16">
        {/* Hero Section */}
        <section className="min-h-screen flex items-center justify-center px-6 relative overflow-hidden" id="hero">
          <div ref={heroRef} className="max-w-4xl text-center parallax-layer" data-speed="0.1">
            <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-[#4edea3]/10 border border-[#4edea3]/20 rounded-full mb-6">
              <Sparkles className="size-4 text-[#4edea3]" />
              <span className="text-[10px] font-bold tracking-widest text-[#4edea3] uppercase">Next-Gen Spreadsheet Intelligence</span>
            </div>
            <h1 className="text-5xl md:text-7xl lg:text-80px leading-[1.1] mb-6 text-white font-bold tracking-tight">
              Excel, <span className="text-[#4edea3] italic">Accelerated</span> <br/>by Intelligence
            </h1>
            <p className="text-base md:text-xl text-[#bbcabf] mb-10 max-w-2xl mx-auto leading-relaxed">
              Meet DataPilot: The AI agent that cleans, analyzes, and transforms your data in seconds. Stop wrestling with cells, start making decisions.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/dashboard" className="w-full sm:w-auto bg-[#4edea3] text-[#003824] px-8 py-3.5 rounded-lg text-sm font-bold hover-lift active:scale-95 text-center">Start Your Free Trial</Link>
              <Link href="/dashboard" className="w-full sm:w-auto glass-card text-white px-8 py-3.5 rounded-lg text-sm font-semibold border border-[#3c4a42] hover:bg-[#2d3449]/20 transition-all text-center">Watch Demo</Link>
            </div>
          </div>
        </section>

        {/* Problem/Solution Section */}
        <section className="py-24 px-6">
          <div className="max-w-7xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
              <div className="lg:col-span-5 parallax-layer" data-speed="-0.05">
                <h2 className="text-3xl md:text-4xl text-white font-bold leading-tight mb-6">
                  Spreadsheets are the backbone of business, but they're <span className="text-[#ffb4ab]">broken</span>.
                </h2>
                <p className="text-[#bbcabf] mb-8 leading-relaxed">
                  Legacy tools weren't built for the scale and complexity of today's data. Manual cleaning takes hours. Complex formulas are prone to error. Insights remain hidden behind technical barriers.
                </p>
                <div className="space-y-4">
                  <div className="flex items-start gap-4 p-4 glass-card rounded-xl">
                    <TimerOff className="size-5 text-[#ffb4ab] shrink-0 mt-0.5" />
                    <div>
                      <h4 className="font-bold text-white text-sm md:text-base">60% Time Wasted</h4>
                      <p className="text-xs text-[#bbcabf] mt-1">Analysts spend most of their time cleaning data rather than analyzing it.</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-4 p-4 glass-card rounded-xl">
                    <AlertCircle className="size-5 text-[#ffb4ab] shrink-0 mt-0.5" />
                    <div>
                      <h4 className="font-bold text-white text-sm md:text-base">Formula Fatigue</h4>
                      <p className="text-xs text-[#bbcabf] mt-1">One misplaced comma in a nested IF statement can break entire models.</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-6 parallax-layer" data-speed="0.05">
                <div className="glass-card p-6 rounded-2xl border-[#4edea3]/30 bg-[#4edea3]/5 hover-lift">
                  <div className="w-12 h-12 bg-[#4edea3]/20 rounded-lg flex items-center justify-center mb-6">
                    <Zap className="size-6 text-[#4edea3]" />
                  </div>
                  <h3 className="text-lg font-bold text-[#4edea3] mb-2">Instant Execution</h3>
                  <p className="text-xs text-[#bbcabf] leading-relaxed">Processes millions of rows in milliseconds using optimized WebGL acceleration.</p>
                </div>
                <div className="glass-card p-6 rounded-2xl hover-lift">
                  <div className="w-12 h-12 bg-[#0566d9]/20 rounded-lg flex items-center justify-center mb-6">
                    <Brain className="size-6 text-[#adc6ff]" />
                  </div>
                  <h3 className="text-lg font-bold text-white mb-2">Neural Mapping</h3>
                  <p className="text-xs text-[#bbcabf] leading-relaxed">Understands the semantic context of your headers and values automatically.</p>
                </div>
                <div className="sm:col-span-2 glass-card p-6 rounded-2xl flex flex-col sm:flex-row items-center gap-6 hover-lift">
                  <div className="w-16 h-16 bg-[#71af97]/20 rounded-full flex items-center justify-center shrink-0">
                    <Bot className="size-8 text-[#95d3ba]" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white mb-2">Autonomous Agency</h3>
                    <p className="text-xs text-[#bbcabf] leading-relaxed">
                      DataPilot doesn't just suggest—it acts. Assign complex multi-step workflows and watch them complete in real-time with full transparency.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Core Features - Bento Grid */}
        <section className="py-24 px-6 bg-[#060e20]/50">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16">
              <h2 className="text-3xl md:text-4xl text-white font-bold mb-2">The Power of a Data Team in One Agent</h2>
              <p className="text-sm text-[#bbcabf] max-w-xl mx-auto">Built for the high-stakes world of enterprise finance and operations.</p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Feature 1: Smart Clean */}
              <div className="glass-card p-6 rounded-2xl flex flex-col justify-between group hover:ai-glow transition-all duration-500 min-h-[320px]">
                <div>
                  <div className="flex justify-between items-start mb-10">
                    <div className="p-2 bg-[#2d3449] rounded-lg">
                      <Brush className="size-5 text-[#4edea3]" />
                    </div>
                    <span className="text-[10px] font-mono text-[#10b981]/60 font-semibold tracking-wider">DP-MODULE-01</span>
                  </div>
                  <h3 className="text-lg font-bold text-white mb-3">Smart Clean</h3>
                  <p className="text-xs text-[#bbcabf] leading-relaxed">Remove duplicates, fix formatting, and standardize inputs with one click. Our AI recognizes inconsistent date formats and typos across 50+ languages.</p>
                </div>
                <div className="mt-6 p-3.5 bg-[#0b1326]/50 rounded-lg border border-[#3c4a42]/30 font-mono text-[10px] overflow-hidden space-y-1">
                  <div className="flex gap-2 text-[#4edea3]">
                    <span>[AI]</span> <span>Analyzing column 'Date'...</span>
                  </div>
                  <div className="flex gap-2 text-[#bbcabf]">
                    <span>[DP]</span> <span className="animate-pulse">Standardizing to ISO 8601...</span>
                  </div>
                </div>
              </div>

              {/* Feature 2: Insight Engine */}
              <div className="glass-card p-6 rounded-2xl flex flex-col justify-between group md:row-span-2 hover:ai-glow transition-all duration-500 min-h-[380px]">
                <div className="h-full flex flex-col">
                  <div className="flex justify-between items-start mb-10">
                    <div className="p-2 bg-[#2d3449] rounded-lg">
                      <Activity className="size-5 text-[#adc6ff]" />
                    </div>
                    <span className="text-[10px] font-mono text-[#adc6ff]/60 font-semibold tracking-wider">DP-CORE-X</span>
                  </div>
                  <h3 className="text-lg font-bold text-white mb-3">Insight Engine</h3>
                  <p className="text-xs text-[#bbcabf] leading-relaxed mb-6">Ask questions in plain English, get instant visualizations. "Show me the correlation between regional sales and marketing spend for Q3."</p>
                  <div className="mt-auto relative space-y-4">
                    <div className="p-4 bg-[#222a3d] rounded-xl border border-[#3c4a42]/50">
                      <div className="text-xs text-[#bbcabf] italic mb-3">"Compare growth vs LY"</div>
                      <div className="h-28 w-full bg-gradient-to-tr from-[#4edea3]/10 to-[#adc6ff]/10 rounded-lg flex items-end justify-between p-2 gap-1.5">
                        <div className="bg-[#4edea3]/40 w-1/6 h-[40%] rounded-sm"></div>
                        <div className="bg-[#4edea3]/40 w-1/6 h-[60%] rounded-sm"></div>
                        <div className="bg-[#4edea3]/40 w-1/6 h-[55%] rounded-sm"></div>
                        <div className="bg-[#4edea3]/40 w-1/6 h-[85%] rounded-sm animate-pulse"></div>
                        <div className="bg-[#adc6ff]/40 w-1/6 h-[30%] rounded-sm"></div>
                        <div className="bg-[#adc6ff]/40 w-1/6 h-[45%] rounded-sm"></div>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      <span className="px-2 py-0.5 bg-[#adc6ff]/10 text-[#adc6ff] text-[9px] rounded border border-[#adc6ff]/20 font-semibold">Trend Analysis</span>
                      <span className="px-2 py-0.5 bg-[#adc6ff]/10 text-[#adc6ff] text-[9px] rounded border border-[#adc6ff]/20 font-semibold">Pivot Gen</span>
                      <span className="px-2 py-0.5 bg-[#adc6ff]/10 text-[#adc6ff] text-[9px] rounded border border-[#adc6ff]/20 font-semibold">Anomaly Detect</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Feature 3: Formula Architect */}
              <div className="glass-card p-6 rounded-2xl flex flex-col justify-between group hover:ai-glow transition-all duration-500 min-h-[320px]">
                <div>
                  <div className="flex justify-between items-start mb-10">
                    <div className="p-2 bg-[#2d3449] rounded-lg">
                      <Binary className="size-5 text-[#95d3ba]" />
                    </div>
                    <span className="text-[10px] font-mono text-[#95d3ba]/60 font-semibold tracking-wider">DP-BUILD-42</span>
                  </div>
                  <h3 className="text-lg font-bold text-white mb-3">Formula Architect</h3>
                  <p className="text-xs text-[#bbcabf] leading-relaxed">Never write a complex nested IF or VLOOKUP again. Describe the logic, and DataPilot generates the optimized, error-free syntax.</p>
                </div>
                <div className="mt-6 font-mono text-[10px] text-[#95d3ba] p-3 bg-[#95d3ba]/5 rounded-lg border border-[#95d3ba]/25 overflow-x-auto">
                  =IF(AND(A2&gt;100, B2="Tier 1"), C2*1.15, C2*0.95)
                </div>
              </div>

              {/* Feature 4: Integration (Wide) */}
              <div className="md:col-span-2 glass-card p-6 rounded-2xl flex flex-col sm:flex-row items-center justify-between gap-6 hover:ai-glow transition-all duration-500 min-h-[160px]">
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-white mb-2">Seamless Ecosystem</h3>
                  <p className="text-xs text-[#bbcabf] leading-relaxed">DataPilot connects directly to your existing tech stack. Export to Excel, Google Sheets, or stream live data via API connectors. No new software to learn.</p>
                </div>
                <div className="flex -space-x-3 shrink-0">
                  <div className="w-12 h-12 rounded-full bg-[#222a3d] border-2 border-[#3c4a42] flex items-center justify-center shadow-lg">
                    <Table2 className="size-5 text-[#4edea3]" />
                  </div>
                  <div className="w-12 h-12 rounded-full bg-[#222a3d] border-2 border-[#3c4a42] flex items-center justify-center shadow-lg">
                    <CloudSync className="size-5 text-[#10b981]" />
                  </div>
                  <div className="w-12 h-12 rounded-full bg-[#222a3d] border-2 border-[#3c4a42] flex items-center justify-center shadow-lg">
                    <Database className="size-5 text-[#adc6ff]" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-24 px-6 relative overflow-hidden">
          <div className="max-w-5xl mx-auto glass-card p-10 md:p-20 rounded-[40px] text-center relative z-10 ai-glow">
            <div className="absolute -top-10 -left-10 w-40 h-40 bg-[#4edea3]/20 blur-[100px] rounded-full"></div>
            <div className="absolute -bottom-10 -right-10 w-40 h-40 bg-[#adc6ff]/20 blur-[100px] rounded-full"></div>
            <h2 className="text-4xl md:text-5xl text-white mb-6 font-bold tracking-tight">
              Ready to outpace the <br/><span className="text-[#4edea3] underline decoration-[#4edea3]/30">manual age?</span>
            </h2>
            <p className="text-base md:text-lg text-[#bbcabf] mb-10 max-w-2xl mx-auto leading-relaxed">Join 10,000+ analysts who have reclaimed 20 hours a week with DataPilot.</p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/dashboard" className="w-full sm:w-auto bg-[#4edea3] text-[#003824] px-8 py-4 rounded-xl text-sm font-bold shadow-[0_0_40px_rgba(78,222,163,0.4)] hover:shadow-[0_0_60px_rgba(78,222,163,0.6)] transition-all active:scale-95 text-center">Start Your Free Trial</Link>
              <div className="flex items-center gap-2 mt-4 sm:mt-0 text-[11px] text-[#bbcabf] font-mono">
                <ShieldCheck className="size-4 text-[#4edea3]" />
                No credit card required
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="bg-[#060e20] border-t border-[#3c4a42]/30 py-16">
          <div className="flex flex-col md:flex-row justify-between items-center px-6 w-full max-w-7xl mx-auto gap-8">
            <div className="flex flex-col items-center md:items-start gap-1.5">
              <span className="font-extrabold text-2xl text-[#4edea3]">DataPilot</span>
              <p className="text-[11px] text-[#bbcabf]">© 2024 DataPilot AI. All rights reserved.</p>
            </div>
            <div className="flex flex-wrap justify-center gap-6 md:gap-10">
              <a className="text-[11px] text-[#bbcabf] hover:text-[#4edea3] transition-colors" href="#">Privacy Policy</a>
              <a className="text-[11px] text-[#bbcabf] hover:text-[#4edea3] transition-colors" href="#">Terms of Service</a>
              <a className="text-[11px] text-[#bbcabf] hover:text-[#4edea3] transition-colors" href="#">Security</a>
              <a className="text-[11px] text-[#bbcabf] hover:text-[#4edea3] transition-colors" href="#">Contact Support</a>
            </div>
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full glass-card flex items-center justify-center cursor-pointer hover:bg-[#4edea3]/10 hover:text-[#4edea3] transition-all">
                <Share2 className="size-4" />
              </div>
              <div className="w-8 h-8 rounded-full glass-card flex items-center justify-center cursor-pointer hover:bg-[#4edea3]/10 hover:text-[#4edea3] transition-all">
                <Globe className="size-4" />
              </div>
            </div>
          </div>
        </footer>
      </main>
    </div>
  )
}
