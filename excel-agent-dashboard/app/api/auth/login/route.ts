import { ENTRA_CONFIG } from "@/lib/auth/config";
import { getMsalApplication, cryptoProvider } from "@/lib/auth/msal-client";
import { setSessionCookie } from "@/lib/auth/session";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const { email, password } = await request.json();

    if (!email || !password) {
      return NextResponse.json({ error: "Email and password are required." }, { status: 400 });
    }

    const isEntraConfigured =
      ENTRA_CONFIG.webClientId !== "placeholder-web-client-id" &&
      !ENTRA_CONFIG.webClientId.includes("placeholder");

    if (isEntraConfigured) {
      const origin = new URL(request.url).origin;
      const redirectUri = `${origin}/api/auth/callback`;

      const msalApp = getMsalApplication();
      const pkceCodes = await cryptoProvider.generatePkceCodes();

      const cookieStore = await cookies();
      cookieStore.set("pkce_verifier", pkceCodes.verifier, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "lax",
        path: "/",
        maxAge: 600, // 10 minutes
      });

      const scopes = ["openid", "profile", "email", "offline_access"];
      if (ENTRA_CONFIG.apiScope && !ENTRA_CONFIG.apiScope.includes("placeholder")) {
        scopes.push(ENTRA_CONFIG.apiScope);
      }

      const authUrl = await msalApp.getAuthCodeUrl({
        scopes,
        redirectUri,
        codeChallenge: pkceCodes.challenge,
        codeChallengeMethod: "S256",
        loginHint: email,
      });

      return NextResponse.json({ redirectUrl: authUrl });
    }

    // Dev / test mode fallback when live Entra External ID credentials are not configured
    const devUserId = `entra_${Buffer.from(email).toString("hex").slice(0, 16)}`;
    await setSessionCookie({
      user_id: devUserId,
      email: email,
      name: email.split("@")[0],
      authProvider: "Microsoft Entra External ID (Dev Mode)",
      accessToken: `dev_token_${devUserId}`,
    });

    return NextResponse.json({ ok: true });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Authentication error" },
      { status: 500 }
    );
  }
}
