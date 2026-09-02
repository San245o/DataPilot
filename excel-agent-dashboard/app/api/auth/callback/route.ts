import { ENTRA_CONFIG } from "@/lib/auth/config";
import { getMsalApplication } from "@/lib/auth/msal-client";
import { setSessionCookie } from "@/lib/auth/session";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const requestUrl = new URL(request.url);
  const code = requestUrl.searchParams.get("code");
  const error = requestUrl.searchParams.get("error");
  const errorDescription = requestUrl.searchParams.get("error_description");

  if (error) {
    console.error("Entra External ID OAuth error:", error, errorDescription);
    return NextResponse.redirect(
      new URL(`/login?error=${encodeURIComponent(errorDescription || error)}`, requestUrl.origin)
    );
  }

  if (!code) {
    return NextResponse.redirect(new URL("/login?error=no_code", requestUrl.origin));
  }

  try {
    const redirectUri = `${requestUrl.origin}/api/auth/callback`;
    const cookieStore = await cookies();
    const codeVerifier = cookieStore.get("pkce_verifier")?.value;

    const msalApp = getMsalApplication();

    const scopes = ["openid", "profile", "email", "offline_access"];
    if (ENTRA_CONFIG.apiScope && !ENTRA_CONFIG.apiScope.includes("placeholder")) {
      scopes.push(ENTRA_CONFIG.apiScope);
    }

    const tokenResponse = await msalApp.acquireTokenByCode({
      code,
      codeVerifier: codeVerifier || undefined,
      redirectUri,
      scopes,
    });

    cookieStore.delete("pkce_verifier");

    const claims = (tokenResponse.idTokenClaims as Record<string, unknown>) || {};
    const userId = (tokenResponse.account?.homeAccountId || claims.oid || claims.sub) as string;
    const email = (tokenResponse.account?.username || claims.email || claims.preferred_username || "user@entra.com") as string;
    const name = (tokenResponse.account?.name || claims.name || email.split("@")[0]) as string;

    await setSessionCookie({
      user_id: userId,
      email: email,
      name: name,
      accessToken: tokenResponse.accessToken, // Access token specifically for DataPilot API
      idToken: tokenResponse.idToken,         // ID token for Web frontend identity
      authProvider: "Microsoft Entra External ID",
    });

    return NextResponse.redirect(new URL("/home", requestUrl.origin));
  } catch (err: unknown) {
    console.error("MSAL Node token exchange error:", err);
    return NextResponse.redirect(new URL("/login?error=token_exchange_failed", requestUrl.origin));
  }
}
