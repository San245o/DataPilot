import { SESSION_COOKIE } from "@/lib/api"
import { NextResponse, type NextRequest } from "next/server"

const backendUrl = (process.env.BACKEND_URL ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "")

function redirect(request: NextRequest, path: string) {
  const url = new URL(path, request.url)
  const host = request.headers.get("host")
  // Next can normalize request.url to localhost even when the browser used 127.0.0.1.
  if (host && /^(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?$/i.test(host)) {
    url.host = host
  }
  return NextResponse.redirect(url)
}

export async function middleware(request: NextRequest) {
  const { pathname, search } = request.nextUrl
  const token = request.cookies.get(SESSION_COOKIE)?.value
  const isLogin = pathname === "/login"
  let signedIn = false

  if (token && /^[A-Za-z0-9_-]{43}$/.test(token)) {
    try {
      const response = await fetch(`${backendUrl}/auth/session`, {
        headers: { Cookie: `${SESSION_COOKIE}=${token}` },
        cache: "no-store",
        signal: AbortSignal.timeout(3000),
      })
      signedIn = response.ok
    } catch {
      // An unavailable backend cannot validate the session.
    }
  }

  if (!isLogin && !signedIn) {
    const params = new URLSearchParams({ redirectTo: pathname + search })
    return redirect(request, `/login?${params}`)
  }

  if (isLogin && signedIn) {
    return redirect(request, "/home")
  }

  return NextResponse.next()
}

export const config = {
  matcher: ["/login", "/dashboard/:path*", "/report/:path*", "/chart-viewer/:path*", "/home/:path*", "/history/:path*", "/tutorials/:path*"],
}
