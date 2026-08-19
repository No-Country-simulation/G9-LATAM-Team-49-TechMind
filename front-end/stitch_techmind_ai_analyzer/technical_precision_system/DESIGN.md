---
name: Technical Precision System
colors:
  surface: '#fcf8fa'
  surface-dim: '#dcd9db'
  surface-bright: '#fcf8fa'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f6f3f5'
  surface-container: '#f0edef'
  surface-container-high: '#eae7e9'
  surface-container-highest: '#e4e2e4'
  on-surface: '#1b1b1d'
  on-surface-variant: '#45464d'
  inverse-surface: '#303032'
  inverse-on-surface: '#f3f0f2'
  outline: '#76777d'
  outline-variant: '#c6c6cd'
  surface-tint: '#565e74'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#131b2e'
  on-primary-container: '#7c839b'
  inverse-primary: '#bec6e0'
  secondary: '#5c5e68'
  on-secondary: '#ffffff'
  secondary-container: '#dedfeb'
  on-secondary-container: '#60626c'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#271901'
  on-tertiary-container: '#98805d'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dae2fd'
  primary-fixed-dim: '#bec6e0'
  on-primary-fixed: '#131b2e'
  on-primary-fixed-variant: '#3f465c'
  secondary-fixed: '#e1e2ed'
  secondary-fixed-dim: '#c4c6d1'
  on-secondary-fixed: '#191b24'
  on-secondary-fixed-variant: '#444650'
  tertiary-fixed: '#fcdeb5'
  tertiary-fixed-dim: '#dec29a'
  on-tertiary-fixed: '#271901'
  on-tertiary-fixed-variant: '#574425'
  background: '#fcf8fa'
  on-background: '#1b1b1d'
  surface-variant: '#e4e2e4'
typography:
  display-lg:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '600'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '500'
    lineHeight: 32px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  base: 4px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  container-max: 1440px
  sidebar-width: 280px
---

## Brand & Style

The design system is rooted in the "Technical Precision" aesthetic—a sophisticated blend of high-end SaaS professionalism and minimalist editorial design. It targets a highly technical audience of analysts, engineers, and researchers who value clarity over decoration. 

The visual narrative is built on the concept of "The Structured Mind." It uses a **Corporate/Modern** style with **Minimalist** leanings, prioritizing information density without sacrificing whitespace. Every element must feel intentional, avoiding clichéd AI imagery (no brains or robots) in favor of abstract data representations, clean geometry, and high-quality typography. The emotional response should be one of quiet confidence, reliability, and intellectual rigor.

## Colors

This design system utilizes a "Cold Professional" palette. The foundation is built on Deep Navy for authority and Slate for content.

- **Primary (Deep Navy):** Reserved for high-level structural elements like the sidebar and primary headers to ground the interface.
- **Accents (Electric Blue & Violet):** Used sparingly to denote interactivity, AI-driven insights, and data highlights. Violet specifically identifies "AI reasoning" or "Analysis" features.
- **Surface Tiers:** Backgrounds transition from the Main page color (#F8FAFC) to a slightly deeper gray (#F1F5F9) for secondary panels or wells, creating subtle hierarchical separation.
- **Status Colors:** Use standard semantic reds and greens, but desaturate them slightly to maintain the sophisticated atmosphere.

## Typography

The system uses a dual-font approach. **Geist** provides a technical, precise feel for headlines and functional labels, while **Inter** ensures maximum readability for long-form analysis and data tables.

- **Headlines:** Use tighter letter-spacing and medium-to-semibold weights to create a "locked-in" technical look.
- **Body Text:** Maintain generous line-height (1.5x) to prevent fatigue during long reading sessions.
- **Labels:** Small labels and tags should use Geist with a slight tracking increase for clarity at small sizes.
- **Monospace (Optional):** For code snippets or raw data strings, use **JetBrains Mono** to complement the technical aesthetic.

## Layout & Spacing

The design system employs a **Fixed Grid** model for the main dashboard and a **Fluid Content** area for analysis reports.

- **Desktop Layout:** A permanent 280px sidebar on the left. The main content area uses a 12-column grid with 24px gutters and 32px outer margins.
- **Rhythm:** All spacing must be multiples of 4px. Use 16px (md) for standard internal component padding and 24px (lg) for spacing between major sections.
- **Breakpoints:**
  - *Desktop:* 1280px+ (Sidebar visible)
  - *Tablet:* 768px - 1279px (Sidebar collapses to icon-only or drawer)
  - *Mobile:* Under 767px (Single column, 16px margins, vertical stacking for cards)

## Elevation & Depth

To maintain a sophisticated and flat professional look, the system avoids heavy drop shadows. Depth is communicated via **Tonal Layers** and **Low-Contrast Outlines**.

- **Level 0 (Background):** #F8FAFC. The canvas.
- **Level 1 (Cards/Panels):** White (#FFFFFF) surface with a 1px border (#E2E8F0). No shadow.
- **Level 2 (Dropdowns/Modals):** White surface with a 1px border and a very soft, diffused ambient shadow (0px 4px 20px rgba(15, 23, 42, 0.05)).
- **Interactive States:** When hovering over an interactive element, use a subtle background shift to #F1F5F9 rather than increasing elevation.

## Shapes

The shape language is "Soft-Technical." We use consistent, small-radius corners to keep the UI feeling modern but grounded.

- **Standard Elements:** Inputs, buttons, and small cards use a 0.25rem (4px) radius.
- **Large Containers:** Content blocks and main dashboard cards use a 0.5rem (8px) radius.
- **Interactive Pills:** Tags and status indicators use a full pill shape (100px) to distinguish them from structural elements.

## Components

### Buttons
- **Primary:** Deep Navy background, white text. No gradient. 4px corner radius.
- **Secondary:** White background, 1px border (#E2E8F0), Slate text.
- **Ghost:** No background or border, Blue or Slate text, used for tertiary actions.

### Cards
Cards are the primary container for technical insights. They feature a white background, 1px #E2E8F0 border, and no shadow. The header of the card should use a subtle 1px bottom border to separate title and content.

### Chips & Tags
- **Data Chips:** Small, 12px Geist font, neutral gray background (#F1F5F9).
- **AI Insights:** Highlighted with a Violet border and very pale violet background tint.

### Input Fields
Strict, rectangular (4px radius) with a 1px #E2E8F0 border. On focus, the border transitions to Electric Blue with a 2px outer "glow" of 10% opacity blue.

### Pipeline Steps
A vertical or horizontal "stepper" component using thin lines and 8px circular nodes. Active steps use Electric Blue; completed steps use Deep Navy; pending steps use Slate.

### Sidebar
The sidebar uses a Deep Navy background with a 10% opacity white overlay for active states. Navigation items use 14px Geist with a subtle icon to the left.