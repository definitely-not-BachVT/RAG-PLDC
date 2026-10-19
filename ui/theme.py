"""Design system: "Vở bài tập và lời phê" (exercise book and the teacher's red pen).

The learner writes in violet ink (mực tím), the grader marks in red ink (mực đỏ),
on cool white ruled paper with a red margin. One motion only: the red circle drawn
around the correct answer when the learner submits.
"""
from nicegui import ui

PAPER = "#FAFBFC"
SHEET = "#FFFFFF"
INK = "#1F2A44"  # blue-black fountain-pen ink: main text
VIOLET = "#6B4FBB"  # mực tím: everything the learner does
RED = "#C62828"  # mực đỏ: everything the grader does (marks, citations)
RULE = "#D6E0EE"  # ruled lines
GRAPHITE = "#667085"  # secondary text

FONTS = ("https://fonts.googleapis.com/css2?"
         "family=Be+Vietnam+Pro:wght@400;500;600"
         "&family=Literata:ital,opsz,wght@0,7..72,400..700;1,7..72,400"
         "&display=swap&subset=vietnamese")

CSS = f"""
:root {{
  --paper: {PAPER}; --sheet: {SHEET}; --ink: {INK}; --violet: {VIOLET}; --red: {RED};
  --rule: {RULE}; --graphite: {GRAPHITE};
  --violet-wash: #F3F0FA; --red-wash: #FCF1F1;
  --serif: 'Literata', Georgia, serif; --sans: 'Be Vietnam Pro', system-ui, sans-serif;
  --margin: 64px;  /* width of the red-ruled margin on the sheet */
}}
body {{ background: var(--paper); color: var(--ink); font-family: var(--sans); font-size: 15px; }}
.nicegui-content {{ padding: 0; gap: 0; }}
:focus-visible {{ outline: 2px solid var(--violet); outline-offset: 2px; }}

/* ---- left rail ------------------------------------------------------------ */
.rail {{ background: var(--paper) !important; border-right: 1px solid var(--rule); }}
.rail .wordmark {{ margin-bottom: .2rem;  font-family: var(--serif); font-weight: 600; font-size: 1.375rem; line-height: 1.2; }}
.rail .tagline {{ color: var(--graphite); font-size: .8125rem; }}
.rail .nav-link {{ display: flex; justify-content: space-between; align-items: center; width: 100%;
  padding: .45rem .6rem; border-radius: 6px; color: var(--ink); text-decoration: none; font-weight: 500; }}
.rail .nav-link:hover {{ background: #EEF1F6; }}
.rail .nav-link.active {{ background: var(--violet-wash); color: var(--violet); }}
.rail .nav-link.soon {{ color: var(--graphite); cursor: default; font-weight: 400; }}
.rail .count {{ font-size: .75rem; color: var(--violet); font-weight: 600; }}
.rail .group-label {{ color: var(--graphite); font-size: .8125rem; margin: 1.1rem 0 .3rem .6rem; }}
.rail .chapter {{ text-align: left; width: 100%; padding: .3rem .6rem; border-radius: 6px; min-height: 0;
  font-size: .875rem; line-height: 1.35; color: var(--ink) !important; font-weight: 400; }}
.rail .chapter:hover {{ background: #EEF1F6; }}
.rail .chapter .q-btn__content {{ justify-content: flex-start; text-align: left; flex-wrap: nowrap; }}
.rail .chapter.active {{ color: var(--violet) !important; background: var(--violet-wash); font-weight: 500; }}
.rail .chapter.link {{ color: var(--violet) !important; padding-left: .6rem; }}
.rail .score {{ font-family: var(--serif); font-size: 1rem; color: var(--ink); }}

/* ---- the sheet (exercise-book page) ------------------------------------------ */
.sheet {{ position: relative; background: var(--sheet); max-width: 820px; width: 100%;
  min-height: 100vh; margin: 0 auto; padding: 2.5rem 2.5rem 4rem calc(var(--margin) + 1.5rem);
  border-left: 1px solid var(--rule); border-right: 1px solid var(--rule); }}
.sheet::before {{ content: ""; position: absolute; top: 0; bottom: 0; left: var(--margin);
  border-left: 2px solid rgba(198, 40, 40, .7); }}
.sheet-head {{ display: flex; justify-content: space-between; align-items: baseline;
  color: var(--graphite); font-size: .875rem; }}
.progress {{ height: 3px; background: var(--rule); margin: .6rem 0 1.75rem; }}
.progress > div {{ height: 100%; background: var(--violet); }}
.question {{ font-family: var(--serif); font-weight: 600; font-size: 1.375rem; line-height: 1.5;
  max-width: 64ch; margin: 0 0 1.25rem; }}

/* ---- options: one ruled line each ---------------------------------------- */
.options {{ border-top: 1px solid var(--rule); }}
.option {{ position: relative; display: flex; gap: .9rem; align-items: flex-start; width: 100%;
  text-align: left; padding: .85rem .5rem; border-bottom: 1px solid var(--rule); background: none;
  font-family: var(--serif); font-size: 1.0625rem; line-height: 1.55; color: var(--ink); cursor: pointer; }}
.option:hover:not(.locked) {{ background: var(--violet-wash); }}
/* graded options use aria-disabled, not [disabled]: Quasar fades [disabled] with a layered !important */
.option.locked {{ cursor: default; }}
.option.correct .letter {{ border-color: var(--red); color: var(--red); }}
.option .letter {{ position: relative; flex: none; width: 1.9rem; height: 1.9rem; border: 1.5px solid var(--ink); border-radius: 50%;
  display: grid; place-items: center; font-size: .95rem; font-weight: 600; margin-top: -.05rem; }}
.option.picked {{ color: var(--violet); }}
.option.picked .letter {{ background: var(--violet); border-color: var(--violet); color: white; }}
.option.struck .text {{ text-decoration: line-through; text-decoration-color: var(--red);
  text-decoration-thickness: 1.5px; }}
.margin-mark {{ position: absolute; left: calc(-1 * var(--margin) - 1.5rem + 18px); top: .7rem; width: 26px;
  height: 26px; color: var(--red); }}

/* the red circle: drawn once, when the learner submits */
.circle {{ position: absolute; left: -11px; top: -9px; width: calc(100% + 22px); height: calc(100% + 18px);
  pointer-events: none; overflow: visible; }}
.circle path {{ fill: none; stroke: var(--red); stroke-width: 2.2; stroke-linecap: round;
  vector-effect: non-scaling-stroke; }}
.circle.draw path {{ stroke-dasharray: 1; stroke-dashoffset: 1; animation: draw .55s ease-out .1s forwards; }}
@keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .circle.draw path {{ animation: none; stroke-dashoffset: 0; }} }}

.verdict {{ font-family: var(--serif); font-style: italic; color: var(--red); font-size: 1.0625rem;
  margin: 1rem 0 0; }}

/* ---- buttons --------------------------------------------------------------- */
.btn {{ font-family: var(--sans); font-weight: 500; border-radius: 6px; padding: .55rem 1.1rem;
  border: 1.5px solid transparent; cursor: pointer; font-size: .9375rem; line-height: 1.2; }}
.btn:disabled {{ opacity: .45; cursor: default; }}
.btn-violet {{ background: var(--violet); color: white; }}
.btn-red {{ background: white; color: var(--red); border-color: var(--red); }}
.btn-red:hover:not(:disabled) {{ background: var(--red-wash); }}
.btn-plain {{ background: none; color: var(--ink); }}
.btn-plain:hover:not(:disabled) {{ background: #EEF1F6; }}
kbd {{ font-family: var(--sans); font-size: .75rem; border: 1px solid var(--rule); border-bottom-width: 2px;
  border-radius: 4px; padding: 0 .3rem; color: var(--graphite); background: white; }}
.hints {{ color: var(--graphite); font-size: .8125rem; display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 2rem; }}

/* ---- the teacher's comment (AI explanation) ------------------------------------ */
.comment {{ margin-top: 1.75rem; padding-left: 1.1rem; border-left: 2px solid var(--red); max-width: 66ch; }}
.comment h3 {{ font-family: var(--serif); color: var(--red); font-weight: 600; font-size: 1.0625rem; margin: 0 0 .4rem; }}
.comment .body {{ font-family: var(--serif); font-size: 1rem; line-height: 1.65; }}
.comment .body h1, .comment .body h2, .comment .body h3, .comment .body h4 {{
  font-family: var(--serif); color: var(--ink); font-size: 1.0625rem; font-weight: 600; margin: 1rem 0 .3rem; }}
.comment .body p {{ margin: .4rem 0; }}
.comment .body hr {{ border: 0; border-top: 1px solid var(--rule); margin: 1rem 0; }}
.cite {{ color: var(--red); font-family: var(--sans); font-weight: 600; font-size: .72em; cursor: pointer;
  padding: 0 .1em; }}
.cite:hover {{ text-decoration: underline; }}
.sources {{ margin-top: .9rem; font-size: .875rem; color: var(--graphite); }}
.sources button {{ color: var(--ink); text-align: left; padding: .15rem 0; }}
.sources button:hover {{ color: var(--red); }}
.sources .num {{ color: var(--red); font-weight: 600; margin-right: .4rem; }}
.notice {{ margin-top: 1rem; padding: .8rem 1rem; border: 1px solid var(--rule); border-radius: 6px;
  background: var(--paper); max-width: 66ch; }}

/* ---- source panel (right drawer) -------------------------------------------- */
.source-panel {{ background: var(--paper) !important; border-left: 1px solid var(--rule); }}
.source-panel .excerpt {{ font-family: var(--serif); font-size: .9375rem; line-height: 1.6;
  max-height: 30vh; overflow: auto; border-left: 2px solid var(--rule); padding-left: .8rem; }}
.source-panel img {{ border: 1px solid var(--rule); background: white; }}

/* ---- summary --------------------------------------------------------------------- */
.headline {{ font-family: var(--serif); font-size: 1.75rem; line-height: 1.35; font-weight: 500; max-width: 30ch; }}
.bars {{ display: grid; grid-template-columns: max-content 1fr max-content; gap: .55rem .9rem;
  align-items: center; margin: 1.5rem 0 2rem; max-width: 560px; }}
.bars .name {{ font-size: .875rem; color: var(--graphite); }}
.bars .track {{ height: 10px; background: var(--rule); border-radius: 0 4px 4px 0; }}
.bars .fill {{ height: 100%; background: var(--violet); border-radius: 0 4px 4px 0; min-width: 2px; }}
.bars .val {{ font-size: .875rem; font-variant-numeric: tabular-nums; }}
.wrong-item {{ border-bottom: 1px solid var(--rule); padding: .8rem 0; font-family: var(--serif); }}
.wrong-item .q {{ font-weight: 600; }}
.wrong-item .a {{ color: var(--graphite); font-size: .9375rem; margin-top: .2rem; }}

.contents {{ display: contents; }}
.mobile-bar {{ display: none; }}
@media (max-width: 900px) {{
  .mobile-bar {{ display: flex; align-items: center; gap: .5rem; padding: .4rem .6rem;
    border-bottom: 1px solid var(--rule); background: var(--paper); position: sticky; top: 0; z-index: 10; }}
}}
@media (max-width: 700px) {{
  :root {{ --margin: 28px; }}
  .sheet {{ padding: 1.5rem 1rem 3rem calc(var(--margin) + .9rem); border: 0; }}
  .question {{ font-size: 1.1875rem; }}
  .hints {{ display: none; }}
  .margin-mark {{ width: 20px; height: 20px; left: calc(-1 * var(--margin) - .9rem + 4px); }}
}}
"""

# Hand-drawn ellipse (a slightly open loop, like a pen circling an answer). Normalized to a
# 100x100 box; pathLength=1 lets the CSS animate it regardless of its real length.
CIRCLE_SVG = ('<svg class="circle{cls}" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">'
              '<path pathLength="1" d="M 8 30 C 14 6, 86 2, 96 34 C 101 62, 78 98, 44 96 '
              'C 14 94, 0 72, 3 48 C 5 32, 20 16, 40 12"/></svg>')
TICK_SVG = ('<svg class="margin-mark" viewBox="0 0 26 26" aria-hidden="true"><path d="M4 14 L10 20 L22 5" '
            'fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>')
CROSS_SVG = ('<svg class="margin-mark" viewBox="0 0 26 26" aria-hidden="true"><path d="M6 6 L20 20 M20 6 L6 20" '
             'fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>')

# Citation markers [n] in AI text are turned into <sup class="cite" data-n="n">; one delegated
# listener forwards clicks to Python as a 'cite' event.
CITE_JS = """
document.addEventListener('click', (e) => {
  const c = e.target.closest('.cite');
  if (c) emitEvent('cite', Number(c.dataset.n));
});
"""


def apply_theme() -> None:
    ui.add_head_html(f'<link rel="preconnect" href="https://fonts.googleapis.com">'
                     f'<link rel="stylesheet" href="{FONTS}"><style>{CSS}</style>'
                     f'<script>{CITE_JS}</script>')
    ui.colors(primary=VIOLET, secondary=INK, accent=RED, negative=RED, positive=VIOLET)
