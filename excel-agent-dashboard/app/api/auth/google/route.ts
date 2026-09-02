import { ENTRA_CONFIG } from "@/lib/auth/config";
import { getMsalApplication, cryptoProvider } from "@/lib/auth/msal-client";
import { setSessionCookie } from "@/lib/auth/session";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const origin = new URL(request.url).origin;

  const isEntraConfigured =
    ENTRA_CONFIG.webClientId !== "placeholder-web-client-id" &&
    !ENTRA_CONFIG.webClientId.includes("placeholder");

  if (isEntraConfigured) {
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
      domainHint: "google.com",
    });

    return NextResponse.redirect(authUrl);
  }

  // Dev mode fallback for testing Google auth without live Entra keys
  const devUserId = "google_entra_dev_user_12345";
  await setSessionCookie({
    user_id: devUserId,
    email: "user@gmail.com",
    name: "Google Entra User",
    authProvider: "Google via Entra External ID (Dev)",
    accessToken: `google_token_${devUserId}`,
  });

  return NextResponse.redirect(new URL("/home", origin));
}
