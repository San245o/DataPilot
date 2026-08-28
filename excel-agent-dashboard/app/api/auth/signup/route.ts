import { ENTRA_CONFIG } from "@/lib/auth/config";
import { getMsalApplication, cryptoProvider } from "@/lib/auth/msal-client";
import { setSessionCookie } from "@/lib/auth/session";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const { fullName, email, password } = await request.json();

    if (!email || !password || !fullName) {
      return NextResponse.json({ error: "All fields are required." }, { status: 400 });
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
        maxAge: 600,
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
        prompt: "create",
      });

      return NextResponse.json({ redirectUrl: authUrl });
    }

    const devUserId = `entra_${Buffer.from(email).toString("hex").slice(0, 16)}`;
    await setSessionCookie({
      user_id: devUserId,
      email: email,
      name: fullName,
      authProvider: "Microsoft Entra External ID (Dev Mode)",
      accessToken: `dev_token_${devUserId}`,
    });

    return NextResponse.json({ ok: true });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Registration error" },
      { status: 500 }
    );
  }
}
