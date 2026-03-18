/**
 * Helpers E2E — mocks API et utilitaires de navigation
 */

import type { Page } from "@playwright/test";

// ─── Réponses API mockées ────────────────────────────────────────────────────

export const MOCK_PREDICT_LIGHT = {
  label: 0,
  score_distress: 0.15,
  model: "baseline",
  text_preview: "je vais bien",
  detected_lang: "fr",
};

export const MOCK_PREDICT_ELEVATED = {
  label: 1,
  score_distress: 0.55,
  model: "baseline",
  text_preview: "je suis stressé",
  detected_lang: "fr",
};

export const MOCK_PREDICT_CRITICAL = {
  label: 1,
  score_distress: 0.90,
  model: "baseline",
  text_preview: "je veux mourir",
  detected_lang: "fr",
};

export const MOCK_SOLUTIONS = {
  level: 2,
  clinicalProfile: "adjustment",
  message: "Ce que vous traversez mérite toute votre attention.",
  closing: "Un pas à la fois.",
  microActions: [
    { id: "breathing", title: "Respiration cohérente", description: "Inspirez 4s, expirez 4s.", duration: "2 min" },
  ],
  therapeuticBrick: "cbt_activation",
  resources: [],
  escalationRequired: false,
};

export const MOCK_SOLUTIONS_CRISIS = {
  level: 4,
  clinicalProfile: "crisis",
  message: "Vous n'êtes pas seul(e). Des personnes sont disponibles maintenant.",
  closing: "Ne restez pas seul(e).",
  microActions: [],
  therapeuticBrick: "crisis",
  resources: [
    { id: "3114", label: "3114", detail: "Numéro national prévention suicide", type: "phone", href: "tel:3114", urgent: true },
  ],
  escalationRequired: true,
};

// ─── Setup mocks réseau ───────────────────────────────────────────────────────

export async function mockApi(page: Page, predictResponse = MOCK_PREDICT_ELEVATED, solutionsResponse = MOCK_SOLUTIONS) {
  await page.route("**/predict", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(predictResponse) })
  );
  await page.route("**/solutions", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(solutionsResponse) })
  );
  // /analyze toujours en 503 dans les tests — le frontend doit dégradation gracieuse
  await page.route("**/analyze", (route) =>
    route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Service indisponible (test)" }) })
  );
}

// ─── Navigation helper ────────────────────────────────────────────────────────

export async function goToEmotions(page: Page, mode: "kids" | "adult" = "adult") {
  await page.goto("/");
  // Choisir le mode
  if (mode === "kids") {
    await page.getByRole("button", { name: /enfant|kids/i }).click();
  } else {
    await page.getByRole("button", { name: /adulte|adult/i }).click();
  }
  await page.waitForURL(/\/emotions/);
}
