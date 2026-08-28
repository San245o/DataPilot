import { getSessionFromCookies } from "@/lib/auth/session";
import { NextResponse } from "next/server";

export async function GET() {
  const session = await getSessionFromCookies();
  if (!session) {
    return NextResponse.json({ user: null }, { status: 401 });
  }

  // Never expose raw access/ID tokens to client JavaScript
  const { accessToken, idToken, ...userClaims } = session;
  return NextResponse.json({ user: userClaims });
}
