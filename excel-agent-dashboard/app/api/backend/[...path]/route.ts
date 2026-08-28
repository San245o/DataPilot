import { getSessionFromCookies } from "@/lib/auth/session";
import { NextResponse, type NextRequest } from "next/server";

const BACKEND_URL = (process.env.BACKEND_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function handleProxy(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  const pathString = path.join("/");
  const targetUrl = `${BACKEND_URL}/${pathString}`;

  const session = await getSessionFromCookies();
  const headers = new Headers(request.headers);

  // Clean host header
  headers.delete("host");

  // Attach Bearer Access Token server-side from HTTP-only session cookie
  if (session?.accessToken) {
    headers.set("Authorization", `Bearer ${session.accessToken}`);
  }

  try {
    const body = request.method !== "GET" && request.method !== "HEAD" ? await request.arrayBuffer() : undefined;

    const backendResponse = await fetch(targetUrl, {
      method: request.method,
      headers,
      body,
    });

    return new NextResponse(backendResponse.body, {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
      headers: backendResponse.headers,
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { detail: err instanceof Error ? err.message : "Backend proxy connection error" },
      { status: 502 }
    );
  }
}

export const GET = handleProxy;
export const POST = handleProxy;
export const PUT = handleProxy;
export const DELETE = handleProxy;
