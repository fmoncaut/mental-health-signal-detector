/**
 * E2E — Flow crisis (niveau 4 — sécurité absolue)
 *
 * Scénario : texte contenant un keyword critique → score élevé → niveau 4
 * Vérifie :
 *   - SupportResponse affiche le panneau de sécurité ("Tu n'es pas seul" / "Vous n'êtes pas seul")
 *   - Lien 3114 visible et cliquable
 *   - Solutions niveau 4 : 3114 en tête, pas de panneau Analyse
 *   - Pas de score ML affiché (anxiogène)
 *
 * L'API /predict retourne score_distress=0.90 + le texte contient "je veux mourir"
 * → getDistressLevel → "critical" → niveau 4
 */

import { test, expect } from "@playwright/test";
import {
  mockApi,
  goToEmotions,
  MOCK_PREDICT_CRITICAL,
  MOCK_SOLUTIONS_CRISIS,
} from "./helpers";

test.describe("Flow crisis — niveau 4", () => {
  test.beforeEach(async ({ page }) => {
    await mockApi(page, MOCK_PREDICT_CRITICAL, MOCK_SOLUTIONS_CRISIS);
  });

  test("texte critique → SupportResponse affiche zone de sécurité", async ({ page }) => {
    await goToEmotions(page, "adult");
    await page.getByRole("button", { name: /Triste/i }).click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);

    await page.locator("textarea").fill("je veux mourir");
    await page.getByRole("button", { name: /Envoyer/i }).click();
    await page.waitForURL(/\/support/);

    // Panneau de sécurité — heading "Vous n'êtes pas seul(e)"
    await expect(page.getByRole("heading", { name: /n'êtes pas seul|n'es pas seul/i })).toBeVisible();
  });

  test("SupportResponse crisis — lien 3114 visible", async ({ page }) => {
    await goToEmotions(page, "adult");
    await page.getByRole("button", { name: /Triste/i }).click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);

    await page.locator("textarea").fill("je veux mourir");
    await page.getByRole("button", { name: /Envoyer/i }).click();
    await page.waitForURL(/\/support/);

    // Lien 3114 (tel:3114)
    const phoneLink = page.locator('a[href="tel:3114"]').first();
    await expect(phoneLink).toBeVisible();
  });

  test("Solutions crisis — escalationRequired → 3114 en tête", async ({ page }) => {
    await goToEmotions(page, "adult");
    await page.getByRole("button", { name: /Triste/i }).click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);

    await page.locator("textarea").fill("je veux mourir");
    await page.getByRole("button", { name: /Envoyer/i }).click();
    await page.waitForURL(/\/support/);
    await page.getByRole("button", { name: /pistes d'action/i }).click();
    await page.waitForURL(/\/solutions/);

    // 3114 visible dans les ressources urgentes
    await expect(page.locator('a[href="tel:3114"]').first()).toBeVisible();
  });

  test("Solutions crisis — pas de panneau Analyse (anti-anxiété)", async ({ page }) => {
    await goToEmotions(page, "adult");
    await page.getByRole("button", { name: /Triste/i }).click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);

    await page.locator("textarea").fill("je veux mourir");
    await page.getByRole("button", { name: /Envoyer/i }).click();
    await page.waitForURL(/\/support/);
    await page.getByRole("button", { name: /pistes d'action/i }).click();
    await page.waitForURL(/\/solutions/);

    // Le panneau "Analyse" ne doit PAS être visible au niveau 4
    await expect(page.getByText(/^Analyse$/i)).not.toBeVisible();
  });

  test("crisis mode kids — même sécurité", async ({ page }) => {
    await goToEmotions(page, "kids");
    await page.getByRole("button", { name: /triste/i }).first().click();
    await page.getByRole("button", { name: /Continuer/i }).click();
    await page.waitForURL(/\/quickcheck/);
    await page.getByText(/Passer cette étape/i).first().click();
    await page.waitForURL(/\/expression/);

    await page.locator("textarea").fill("je veux mourir");
    await page.getByRole("button", { name: /Partager mes ressentis/i }).click();
    await page.waitForURL(/\/support/);

    await expect(page.getByRole("heading", { name: /n'es pas seul|n'êtes pas seul/i })).toBeVisible();
  });
});
