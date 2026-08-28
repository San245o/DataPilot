import { deleteSessionCookie } from "@/lib/auth/session";
import { NextResponse } from "next/server";

export async function POST() {
  await deleteSessionCookie();
  return NextResponse.json({ ok: true });
}
