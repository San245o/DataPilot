import { decryptSession, COOKIE_NAME } from "@/lib/auth/session";
import { NextResponse, type NextRequest } from "next/server";

function isDevAuthBypassActive(): boolean {
  return (
    process.env.NODE_ENV !== "production" &&
    process.env.DEV_AUTH_BYPASS === "true"
  );
}

export async function middleware(request: NextRequest) {
  const pathname = request.nextUrl.pathname;

  const isProtectedRoute =
    pathname.startsWith("/dashboard") ||
    pathname.startsWith("/report") ||
    pathname.startsWith("/chart-viewer");

  const isAuthRoute =
    pathname.startsWith("/login") ||
    pathname.startsWith("/register") ||
    pathname.startsWith("/signup");

  // When development bypass is active (NODE_ENV !== "production" && DEV_AUTH_BYPASS === "true")
  if (isDevAuthBypassActive()) {
    return NextResponse.next();
  }

  // Normal Entra authentication route protection
  const sessionToken = request.cookies.get(COOKIE_NAME)?.value;
  const session = sessionToken ? await decryptSession(sessionToken) : null;

  if (isProtectedRoute && !session) {
    const redirectUrl = request.nextUrl.clone();
    redirectUrl.pathname = "/login";
    redirectUrl.searchParams.set("redirectTo", pathname);
    return NextResponse.redirect(redirectUrl);
  }

  if (isAuthRoute && session) {
    const redirectUrl = request.nextUrl.clone();
    redirectUrl.pathname = "/dashboard";
    return NextResponse.redirect(redirectUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public files (e.g. demo-poster.jpg, datapilot-demo.mp4)
     */
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|mp4)$).*)",
  ],
};
