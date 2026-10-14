# Accessibility Audit: Project Figures

**Standard:** WCAG 2.1 AA (the applicable parts) · **Scope:** the generated figures in
`reports/figures/` and how they are embedded in the report PDF · **Date:** 2026-10-15

The notebooks produce **static chart images**, so the interactive criteria (keyboard, focus
order, name/role/value) do not apply to the charts themselves — they apply to how the charts are
placed in the report. The criteria that *do* apply to data figures are non-text contrast
(1.4.11), use of colour (1.4.1), and text alternatives (1.1.1). The figures were designed against
these from the start via `src/viz.py`.

## Summary

**Issues found:** 3 | **Critical:** 0 | **Major:** 0 | **Minor:** 3 — all mitigated.

## Findings

| # | Area | Issue | WCAG | Severity | Resolution |
|---|---|---|---|---|---|
| 1 | Use of colour | Two models / three classes distinguished on charts | 1.4.1 | 🟢 Minor | Never colour-alone: confusion matrices annotated with counts; line charts carry distinct markers (●/■/▲) and legends; grouped bars are value-labelled and position-encoded |
| 2 | Non-text contrast | `orange` (2.25:1) and `sky` (2.31:1) fall below 3:1 against white | 1.4.11 | 🟢 Minor | Used only where a text value label or a fixed axis position also carries the meaning, so no information depends on those fills alone |
| 3 | Text alternative | Chart images need alt text when placed in the PDF | 1.1.1 | 🟢 Minor | Every figure has a descriptive title baked in; **action for the report:** add a one-line caption/alt text per figure stating its finding |

## Colour-contrast check (palette vs white, Okabe–Ito)

| Colour | Hex | Ratio | ≥3:1 graphic | Role |
|---|---|---|---|---|
| blue | `#0072B2` | 5.19:1 | ✅ | models/lines, "lowers odds" |
| vermillion | `#D55E00` | 3.87:1 | ✅ | models/lines, "raises odds", Dropout |
| green | `#009E73` | 3.42:1 | ✅ | F1 line, RF importance |
| purple | `#CC79A7` | 3.06:1 | ✅ | spare categorical |
| orange | `#E69F00` | 2.25:1 | ⚠️ (labelled) | Enrolled, PCA bars — always value-labelled |
| sky | `#56B4E9` | 2.31:1 | ⚠️ (labelled) | spare categorical |

## Colour-vision-deficiency safety

The palette is the **Okabe–Ito** set, chosen because it stays distinguishable under deuteranopia,
protanopia and tritanopia. The three outcome classes keep **one fixed colour everywhere**
(Dropout = vermillion, Enrolled = orange, Graduate = blue), so a reader learns the mapping once.
The signed-coefficient charts pair dark **blue** (lowers risk) against mid **vermillion** (raises
risk) — separated by lightness as well as hue, so the direction survives greyscale printing.

## Priority actions for the report

1. **Add a one-sentence caption under each figure** stating what it shows and its finding — this
   doubles as the alt text and satisfies 1.1.1.
2. **Check greyscale** — if the report may be printed in black and white, the lightness ordering
   already keeps the blue/vermillion pairs legible; confirm on a test print.
3. No colour is the sole carrier of meaning anywhere, so the charts remain readable for
   colour-blind and low-vision readers as delivered.
