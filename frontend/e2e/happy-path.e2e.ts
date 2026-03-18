/**
 * E2E — Flow complet adulte et enfant (happy path)
 *
 * Scénario adulte :
 *   Welcome → Mode Adulte → EmotionSelection (stress) → QuickCheck (bridge + passer)
 *   → Expression (texte) → SupportResponse → Solutions
 *
 * Scénario enfant :
 *   Welcome → Mode Enfant → EmotionSelection (tristesse/sadness) → QuickCheck (bridge + passer)
 *   → Expression → SupportResponse → Solutions
 *
 * L'API backend est entièrement mockée — aucune dépendance réseau réelle.
 */

import { test, expect } from "@playwright/test";
import {
  mockApi,
  goToEmotions,
  MOCK_PREDICT_ELEVATED,
  MOCK_SOLUTIONS,
} from "./helpers";

// ─── Flow adulte ──────────────────────────────────────────────────────────────

test.describe("Happy path — mode adulte", () => {
  test.beforeEach(async ({ page }) => {
    await mockApi(page, MOCK_PREDICT_ELEVATED, MOCK_SOLUTIONS);
  });

  test("Welcome affiche les deux modes", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("button", { name: /Mode Enfant/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /Mode Adulte/i })).toBeVisible();
  });

  test("sélection Mode Adulte → EmotionSelection", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /Mode Adulte/i }).click();
    await page.waitForURL(/\/emotions/);
    await expect(page.getByText(/Comment vous sentez-vous/i)).toBeVisible();
  });

  test("sélection émotion stress → QuickCheck bridge", async ({ page }) => {
    await goToEmotions(page, "adult");
    await page.getByRole("button", { name: /Stressé/i }).click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    // Phase bridge : message empathique + bouton Commencer
    await expect(page.getByRole("button", { name: /Commencer/i })).toBeVisible();
  });

  test("QuickCheck → passer → Expression", async ({ page }) => {
    await goToEmotions(page, "adult");
    await page.getByRole("button", { name: /Stressé/i }).click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);
    await expect(page.locator("textarea")).toBeVisible();
  });

  test("QuickCheck → répondre aux questions → Expression", async ({ page }) => {
    await goToEmotions(page, "adult");
    await page.getByRole("button", { name: /Stressé/i }).click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByRole("button", { name: /Commencer/i }).click();
    // 3 questions — répondre à chacune
    for (let i = 0; i < 3; i++) {
      await page.getByRole("radio").first().click();
    }
    await page.waitForURL(/\/expression/);
  });

  test("Expression → envoi → SupportResponse", async ({ page }) => {
    await goToEmotions(page, "adult");
    await page.getByRole("button", { name: /Stressé/i }).click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);

    await page.locator("textarea").fill("je suis stressé par mon travail");
    await page.getByRole("button", { name: /Envoyer/i }).click();
    await page.waitForURL(/\/support/);
    await expect(page.getByText(/Message de soutien/i)).toBeVisible();
  });

  test("SupportResponse → Solutions", async ({ page }) => {
    await goToEmotions(page, "adult");
    await page.getByRole("button", { name: /Stressé/i }).click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);

    await page.locator("textarea").fill("je suis stressé par mon travail");
    await page.getByRole("button", { name: /Envoyer/i }).click();
    await page.waitForURL(/\/support/);
    await page.getByRole("button", { name: /pistes d'action/i }).click();
    await page.waitForURL(/\/solutions/);
    await expect(page.getByText(/Actions concrètes|pistes d'action|Vos pistes/i)).toBeVisible();
  });
});

// ─── Flow enfant ──────────────────────────────────────────────────────────────

test.describe("Happy path — mode enfant", () => {
  test.beforeEach(async ({ page }) => {
    await mockApi(page, MOCK_PREDICT_ELEVATED, {
      ...MOCK_SOLUTIONS,
      level: 2,
    });
  });

  test("sélection Mode Enfant → EmotionSelection", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: /Mode Enfant/i }).click();
    await page.waitForURL(/\/emotions/);
    await expect(page.getByText(/Comment tu te sens/i)).toBeVisible();
  });

  test("sélection émotion triste → QuickCheck bridge kids", async ({ page }) => {
    await goToEmotions(page, "kids");
    // En mode kids les labels sont différents : "Un peu triste"
    await page.getByRole("button", { name: /triste/i }).first().click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await expect(page.getByRole("button", { name: /C'est parti/i })).toBeVisible();
  });

  test("QuickCheck kids → passer → Expression", async ({ page }) => {
    await goToEmotions(page, "kids");
    await page.getByRole("button", { name: /triste/i }).first().click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);
  });

  test("Expression kids → SupportResponse", async ({ page }) => {
    await goToEmotions(page, "kids");
    await page.getByRole("button", { name: /triste/i }).first().click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);

    await page.locator("textarea").fill("je me sens triste");
    await page.getByRole("button", { name: /Partager mes ressentis/i }).click();
    await page.waitForURL(/\/support/);
    await expect(page.getByText(/Message pour toi/i)).toBeVisible();
  });

  test("SupportResponse kids → Solutions kids", async ({ page }) => {
    await goToEmotions(page, "kids");
    await page.getByRole("button", { name: /triste/i }).first().click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);

    await page.locator("textarea").fill("je me sens triste");
    await page.getByRole("button", { name: /Partager mes ressentis/i }).click();
    await page.waitForURL(/\/support/);
    await page.getByRole("button", { name: /Voir ce qui peut t'aider/i }).click();
    await page.waitForURL(/\/solutions/);
    await expect(page.getByText(/Ce que tu peux faire maintenant/i)).toBeVisible();
  });
});

// ─── Émotions positives — bypass QuickCheck ───────────────────────────────────

test.describe("Émotions positives — bypass QuickCheck", () => {
  test("joy → bypass direct vers Expression (pas de QuickCheck)", async ({ page }) => {
    await mockApi(page);
    await goToEmotions(page, "adult");
    await page.getByRole("button", { name: /Joyeux/i }).click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    // QuickCheck détecte joy et redirige directement vers /expression
    await page.waitForURL(/\/expression/);
    await expect(page.locator("textarea")).toBeVisible();
  });
});
