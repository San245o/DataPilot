import { SignJWT, jwtVerify } from "jose";
import { cookies } from "next/headers";
import { ENTRA_CONFIG, isDevAuthBypass } from "./config";

const COOKIE_NAME = "datapilot_session";
const key = new TextEncoder().encode(ENTRA_CONFIG.authSecret);

export type UserSession = {
  user_id: string;
  email: string;
  name?: string;
  accessToken?: string;
  idToken?: string;
  authProvider?: string;
};

export const DEV_USER_IDENTITY: UserSession = {
  user_id: "dev-local-user",
  email: "dev@datapilot.local",
  name: "Local Developer",
  authProvider: "Development Auth Bypass",
  accessToken: "dev_token_dev-local-user",
};

export async function encryptSession(payload: UserSession): Promise<string> {
  return new SignJWT({ ...payload })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("7d")
    .sign(key);
}

export async function decryptSession(token: string): Promise<UserSession | null> {
  try {
    const { payload } = await jwtVerify(token, key, {
      algorithms: ["HS256"],
    });
    return payload as unknown as UserSession;
  } catch {
    return null;
  }
}

export async function getSessionFromCookies(): Promise<UserSession | null> {
  if (isDevAuthBypass()) {
    return DEV_USER_IDENTITY;
  }
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get(COOKIE_NAME)?.value;
  if (!sessionToken) return null;
  return decryptSession(sessionToken);
}

export async function setSessionCookie(session: UserSession): Promise<void> {
  const cookieStore = await cookies();
  const token = await encryptSession(session);

  cookieStore.set(COOKIE_NAME, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 7 * 24 * 60 * 60, // 7 days
  });
}

export async function deleteSessionCookie(): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.delete(COOKIE_NAME);
}

export { COOKIE_NAME };
