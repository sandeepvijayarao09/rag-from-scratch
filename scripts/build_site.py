"""Generate the static /RAG guide site from this repo's markdown.

Output is plain HTML that reuses the design tokens from the personal site's
style.css, so the guide reads as part of that site rather than a bolted-on
docs tool. No Jekyll, no build action: GitHub Pages serves the files directly.

    python scripts/build_site.py <output-dir>

Regenerate after editing any README and the site stays in sync.
"""

import html
import re
import sys
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parents[1]
MD = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "toc"],
                       extension_configs={"codehilite": {"noclasses": False,
                                                         "guess_lang": False}})

NAV_ITEMS = [("Home", "index.html"), ("About", "about.html"), ("Skills", "skills.html"),
             ("Experience", "experience.html"), ("Projects", "projects.html"),
             ("Awards", "awards.html"), ("Contact", "contact.html")]


def nav(depth: str = "../") -> str:
    links = "\n".join(f'          <li><a href="{depth}{h}">{t}</a></li>'
                      for t, h in NAV_ITEMS)
    drawer = "\n".join(f'      <a href="{depth}{h}">{t}</a>' for t, h in NAV_ITEMS)
    return f"""  <header class="nav" role="banner">
    <div class="nav__inner">
      <a class="nav__logo" href="{depth}index.html">Sandeep <span>V.</span></a>
      <nav aria-label="Primary">
        <ul class="nav__links">
{links}
        </ul>
      </nav>
      <a class="nav__cta" href="{depth}contact.html">Get in touch</a>
      <button class="nav__burger" aria-label="Toggle menu" aria-expanded="false">
        <span></span><span></span><span></span>
      </button>
    </div>
    <nav class="nav__drawer" aria-label="Mobile">
{drawer}
      <a class="nav__cta" href="{depth}contact.html">Get in touch</a>
    </nav>
  </header>"""


def footer(depth: str = "../") -> str:
    links = "\n".join(f'          <a href="{depth}{h}">{t}</a>' for t, h in NAV_ITEMS[1:])
    return f"""  <footer class="footer">
    <div class="container">
      <div class="footer__inner">
        <div>
          <div class="footer__brand">Sandeep Vijayarao</div>
          <div class="footer__tagline">Full-stack AI Engineer &middot; San Jose, CA</div>
        </div>
        <nav class="footer__links" aria-label="Footer">
{links}
        </nav>
      </div>
    </div>
  </footer>"""


def page(title: str, body: str, desc: str, depth: str = "../",
         toc: str = "", subnav: str = "") -> str:
    wide = "" if toc else " rg-doc__wrap--wide"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(title)}</title>
  <meta name="description" content="{html.escape(desc)}" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="{depth}style.css" />
  <link rel="stylesheet" href="rag.css" />
</head>
<body>
{nav(depth)}
  <main class="rg-doc">
    <div class="rg-doc__wrap{wide}">
{toc}
      <article class="rg-doc__body">
{subnav}
{body}
      </article>
    </div>
  </main>
{footer(depth)}
  <script src="{depth}main.js"></script>
</body>
</html>
"""


def render(md_text: str) -> tuple[str, str]:
    MD.reset()
    body = MD.convert(md_text)
    return body, getattr(MD, "toc", "")


def strip_h1(html_body: str) -> tuple[str, str]:
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html_body, re.S)
    title = re.sub(r"<.*?>", "", m.group(1)).strip() if m else ""
    # The page already shows the level number in a badge and a breadcrumb, so
    # "Stage 7, Cross-encoder reranking" reads as a stutter. Keep the subject.
    title = re.sub(r"^Stage\s*\d+\s*[,\u2014\u2013-]\s*", "", title)
    return re.sub(r"<h1[^>]*>.*?</h1>", "", html_body, count=1, flags=re.S), title


def fix_links(body: str) -> str:
    """Rewrite in-repo markdown links to the generated page names."""
    body = re.sub(r'href="\.\./\.\./([A-Za-z_]+\.md)"', lambda m: f'href="{m.group(1)[:-3].lower()}.html"', body)
    body = re.sub(r'href="([A-Z][A-Za-z_]*)\.md"', lambda m: f'href="{m.group(1).lower()}.html"', body)
    body = re.sub(r'href="\.\./([0-9]{2})_([a-z_]+)/"', r'href="level-\1.html"', body)
    body = re.sub(r'href="([0-9]{2})_([a-z_]+)/"', r'href="level-\1.html"', body)
    # anything still pointing at repo internals goes to GitHub
    body = re.sub(r'href="(\.\./)*((rag|concepts|scripts)/[^"]+)"',
                  r'href="https://github.com/sandeepvijayarao09/rag-from-scratch/blob/main/\2"', body)
    return body


def stage_dirs() -> list[Path]:
    return sorted((REPO / "concepts").glob("[0-9][0-9]_*"))


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "rag.css").write_text(CSS)

    stages = stage_dirs()
    meta = []
    for d in stages:
        num = d.name[:2]
        readme = d / "README.md"
        if not readme.exists():
            continue
        body, _ = render(readme.read_text())
        body, title = strip_h1(body)
        meta.append((num, d.name, title or d.name, fix_links(body)))

    # per-level pages with prev/next
    for i, (num, dirname, title, body) in enumerate(meta):
        prev_l = f'<a class="rg-pager__prev" href="level-{meta[i-1][0]}.html">&larr; {html.escape(meta[i-1][2])}</a>' if i else '<a class="rg-pager__prev" href="index.html">&larr; Overview</a>'
        next_l = f'<a class="rg-pager__next" href="level-{meta[i+1][0]}.html">{html.escape(meta[i+1][2])} &rarr;</a>' if i + 1 < len(meta) else ''
        crumb = (f'<div class="rg-doc__crumb"><a href="index.html">RAG Guide</a> '
                 f'<span>/</span> Level {int(num)}</div>')
        sub = f'{crumb}<h1 class="rg-doc__title">{html.escape(title)}</h1>'
        sidebar = level_sidebar(meta, num)
        page_html = page(f"{title} — RAG Guide", body + f'<div class="rg-pager">{prev_l}{next_l}</div>',
                         f"Level {int(num)} of a measured 0-to-N guide to building RAG systems.",
                         toc=sidebar, subnav=sub)
        (out / f"level-{num}.html").write_text(page_html)

    # long-form docs
    for src, name, desc in [("LEARN.md", "learn", "A guided path through building RAG systems, basic to advanced."),
                            ("FLOW.md", "flow", "A decision procedure for building RAG from 0 to 1."),
                            ("SYSTEMS.md", "systems", "Taxonomy of RAG systems, organised by the problem each solves."),
                            ("NOTES.md", "notes", "Lab notebook: what broke and what I got wrong.")]:
        p = REPO / src
        if not p.exists():
            continue
        body, toc = render(p.read_text())
        body, title = strip_h1(body)
        body = fix_links(body)
        sidebar = f'<aside class="rg-doc__toc"><div class="rg-doc__toc-title">On this page</div>{toc}</aside>'
        (out / f"{name}.html").write_text(
            page(f"{title} — RAG Guide", body, desc, toc=sidebar,
                 subnav=f'<div class="rg-doc__crumb"><a href="index.html">RAG Guide</a> <span>/</span> {html.escape(title)}</div>'
                        f'<h1 class="rg-doc__title">{html.escape(title)}</h1>'))

    (out / "index.html").write_text(build_index(meta))
    print(f"built {len(meta)} level pages + 5 docs -> {out}")


def level_sidebar(meta, current) -> str:
    items = "\n".join(
        f'<li class="{"rg-is-current" if num == current else ""}">'
        f'<a href="level-{num}.html"><span class="rg-lv">{int(num)}</span> {html.escape(t)}</a></li>'
        for num, _, t, _ in meta)
    return (f'<aside class="rg-doc__toc"><div class="rg-doc__toc-title">Levels</div>'
            f'<ul class="rg-lvlist">{items}</ul></aside>')


TIERS = [("Beginner", "Make it work, then make it measurable", ["01", "02", "03"]),
         ("Core retrieval", "The fundamentals", ["04", "05", "06"]),
         ("Intermediate", "Two-stage and query-side", ["07", "08"]),
         ("Advanced", "Representation and grounding", ["09", "10", "11"]),
         ("Pro", "Control flow and knowledge structure", ["12", "13", "14", "15"])]

RESULTS = [("Chunking", "6 configs, every one", "-0.010 to -0.020", "bad"),
           ("RRF fusion", "4 variants, every one", "-0.009 to -0.024", "bad"),
           ("Reranking on dense", "", "-0.033", "bad"),
           ("HyDE", "", "-0.048", "bad"),
           ("Step-back prompting", "", "-0.066", "bad"),
           ("Query decomposition", "", "-0.095", "bad"),
           ("CRAG relevance grading", "balanced accuracy 0.500", "chance", "bad"),
           ("Weighted hybrid 0.7/0.3", "", "+0.010", "good"),
           ("Late interaction", "at 241x the index", "+0.011", "good"),
           ("Reranking on BM25", "", "+0.041", "good")]


def build_index(meta) -> str:
    titles = {num: t for num, _, t, _ in meta}
    done = {"01", "02", "03", "04", "05", "06", "07", "08", "10", "11", "12"}

    tiers_html = ""
    for name, sub, nums in TIERS:
        rows = ""
        for n in nums:
            if n not in titles:
                continue
            state = "done" if n in done else "wip"
            label = "measured" if n in done else "in progress"
            rows += (f'<a class="rg-lvcard rg-lvcard--{state}" href="level-{n}.html">'
                     f'<span class="rg-lvcard__n">{int(n)}</span>'
                     f'<span class="rg-lvcard__t">{html.escape(titles[n])}</span>'
                     f'<span class="rg-lvcard__s">{label}</span></a>')
        tiers_html += (f'<section class="rg-tier"><h3 class="rg-tier__h">{name}'
                       f'<span>{sub}</span></h3><div class="rg-tier__grid">{rows}</div></section>')

    res = "".join(
        f'<tr class="rg-r--{k}"><td>{html.escape(t)}'
        + (f' <span class="rg-muted">{html.escape(s)}</span>' if s else '')
        + f'</td><td class="rg-num">{d}</td></tr>'
        for t, s, d, k in RESULTS)

    body = f"""
      <div class="rg-hero">
        <div class="rg-hero__eyebrow">Project guide &middot; Level 0 to 15</div>
        <h1 class="rg-hero__h">Building RAG from scratch,<br />and measuring what actually helps</h1>
        <p class="rg-hero__p">I implemented ten recommended RAG techniques and measured each one
        against the same benchmark. <strong>Eight made retrieval worse.</strong> This guide is
        the code, the numbers, and the reasoning for why.</p>
        <div class="rg-hero__meta">
          <div><span>Corpus</span>BEIR SciFact &middot; 5,183 abstracts</div>
          <div><span>Queries</span>300 labelled</div>
          <div><span>Metric</span>nDCG@10</div>
          <div><span>Baseline</span>0.759, validated vs published 0.741</div>
        </div>
        <div class="rg-hero__cta">
          <a class="rg-btn rg-btn--primary" href="learn.html">Start the guide</a>
          <a class="rg-btn" href="https://github.com/sandeepvijayarao09/rag-from-scratch">View the code</a>
        </div>
      </div>

      <section class="rg-panel">
        <h2>What I measured</h2>
        <table class="rg-restable"><tbody>{res}</tbody></table>
        <p class="rg-note">The two clear wins are the two where I measured the gap <em>before</em>
        applying the technique. None of this means these techniques are bad. It means they are
        <strong>conditional</strong>, and the condition is almost always whether your pipeline is
        already good at the thing the technique fixes. Reranking flipped sign inside this repo:
        &minus;0.033 on a strong first stage, +0.041 on a weak one.</p>
      </section>

      <section class="rg-panel">
        <h2>The levels</h2>
        <p class="rg-note">Each level is a distinct kind of RAG system, ordered so every level only
        needs what came before it. Code and a written finding for each.</p>
        {tiers_html}
      </section>

      <section class="rg-panel">
        <h2>Read it a different way</h2>
        <div class="rg-cards">
          <a class="rg-card" href="learn.html"><h4>Learner's guide</h4>
            <p>Eight parts, basic to advanced. Explains each mechanism before showing the result.
            Something to run at every step.</p></a>
          <a class="rg-card" href="flow.html"><h4>Decision procedure</h4>
            <p>The same material as a build order, with a diagnostic gate before each technique.
            For when you already know the techniques.</p></a>
          <a class="rg-card" href="systems.html"><h4>Taxonomy</h4>
            <p>Every kind of RAG system, organised by the problem it solves rather than by
            popularity.</p></a>
          <a class="rg-card" href="notes.html"><h4>Lab notebook</h4>
            <p>What broke, what I got wrong, and the open questions. Including two hypotheses I
            falsified before landing the reranking rule.</p></a>
        </div>
      </section>
"""
    return page("RAG from Scratch — a measured guide, level 0 to 15", body,
                "A measured, level-by-level guide to building RAG systems. Ten techniques "
                "tested on BEIR SciFact; eight made retrieval worse.",
                subnav="")


CSS = """/* RAG guide — extends the site's design tokens */
.rg-doc { padding: calc(var(--nav-h) + 40px) 0 80px; }
.rg-doc__wrap { max-width: 1180px; margin: 0 auto; padding: 0 24px;
  display: grid; grid-template-columns: 240px minmax(0,1fr); gap: 56px; align-items: start; }
.rg-doc__wrap--wide { grid-template-columns: minmax(0,1fr); max-width: 900px; }
.rg-doc__body { min-width: 0; font-size: 16.5px; line-height: 1.72; color: var(--text-primary); }
.rg-doc__crumb { font-size: 13px; color: var(--text-tertiary); margin-bottom: 10px; letter-spacing: .01em; }
.rg-doc__crumb a { color: var(--accent); }
.rg-doc__crumb span { margin: 0 6px; opacity: .5; }
.rg-doc__title { font-size: 40px; line-height: 1.12; letter-spacing: -0.022em; margin: 0 0 28px; font-weight: 650; }

.rg-doc__toc { position: sticky; top: calc(var(--nav-h) + 24px); font-size: 13.5px; }
.rg-doc__toc-title { font-weight: 600; font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--text-tertiary); margin-bottom: 12px; }
.rg-doc__toc ul { list-style: none; margin: 0; padding: 0; }
.rg-doc__toc li { margin: 0; }
.rg-doc__toc a { display: block; padding: 5px 10px; border-radius: 7px; color: var(--text-secondary);
  line-height: 1.45; transition: var(--transition); }
.rg-doc__toc a:hover { background: var(--bg-secondary); color: var(--text-primary); }
.rg-doc__toc > ul > li > ul { margin-left: 10px; border-left: 1px solid var(--border); padding-left: 6px; }
.rg-lvlist .rg-lv { display: inline-block; width: 20px; color: var(--text-tertiary); font-variant-numeric: tabular-nums; }
.rg-lvlist .rg-is-current > a { background: var(--bg-secondary); color: var(--accent); font-weight: 550; }

.rg-doc__body h2 { font-size: 26px; letter-spacing: -0.015em; margin: 44px 0 14px; font-weight: 640;
  padding-top: 14px; border-top: 1px solid var(--border); }
.rg-doc__body h3 { font-size: 19px; margin: 30px 0 10px; font-weight: 620; }
.rg-doc__body p { margin: 0 0 16px; }
.rg-doc__body ul, .rg-doc__body ol { margin: 0 0 18px; padding-left: 22px; }
.rg-doc__body li { margin-bottom: 7px; }
.rg-doc__body p a, .rg-doc__body li a, .rg-doc__body td a,
.rg-doc__body blockquote a { color: var(--accent); }
.rg-doc__body p a:hover, .rg-doc__body li a:hover,
.rg-doc__body td a:hover { text-decoration: underline; }
.rg-doc__body .rg-btn--primary { color: #fff; }
.rg-doc__body .rg-card, .rg-doc__body .rg-lvcard { color: var(--text-primary); }
.rg-doc__body strong { font-weight: 620; }
.rg-doc__body blockquote { margin: 22px 0; padding: 14px 20px; border-left: 3px solid var(--accent);
  background: var(--bg-tertiary); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; color: var(--text-secondary); }
.rg-doc__body blockquote p:last-child { margin-bottom: 0; }

.rg-doc__body pre { background: #1d1d1f; color: #f5f5f7; padding: 18px 20px; border-radius: var(--radius-sm);
  overflow-x: auto; margin: 0 0 20px; font-size: 13px; line-height: 1.62;
  font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
.rg-doc__body code { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.875em; background: var(--bg-secondary); padding: 2px 6px; border-radius: 5px; }
.rg-doc__body pre code { background: none; padding: 0; font-size: inherit; color: inherit; }

.rg-doc__body table { width: 100%; border-collapse: collapse; margin: 0 0 22px; font-size: 14.5px; }
.rg-doc__body th { text-align: left; font-weight: 600; font-size: 12px; letter-spacing: .05em;
  text-transform: uppercase; color: var(--text-tertiary); padding: 8px 12px; border-bottom: 1px solid var(--border-strong); }
.rg-doc__body td { padding: 9px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
.rg-doc__body tr:last-child td { border-bottom: none; }

.rg-pager { display: flex; justify-content: space-between; gap: 16px; margin-top: 56px;
  padding-top: 24px; border-top: 1px solid var(--border); font-size: 14.5px; }
.rg-pager a { color: var(--accent); }
.rg-pager__next { margin-left: auto; text-align: right; }

/* index */
.rg-hero { padding: 8px 0 40px; }
.rg-hero__eyebrow { font-size: 12px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--accent); font-weight: 600; margin-bottom: 14px; }
.rg-hero__h { font-size: 52px; line-height: 1.08; letter-spacing: -0.028em; font-weight: 680; margin: 0 0 20px; }
.rg-hero__p { font-size: 19px; line-height: 1.6; color: var(--text-secondary); max-width: 660px; margin: 0 0 28px; }
.rg-hero__p strong { color: var(--text-primary); font-weight: 620; }
.rg-hero__meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 1px;
  background: var(--border); border: 1px solid var(--border); border-radius: var(--radius-md);
  overflow: hidden; margin-bottom: 30px; }
.rg-hero__meta div { background: var(--bg); padding: 14px 16px; font-size: 14px; font-weight: 550; }
.rg-hero__meta span { display: block; font-size: 11px; letter-spacing: .07em; text-transform: uppercase;
  color: var(--text-tertiary); font-weight: 600; margin-bottom: 5px; }
.rg-hero__cta { display: flex; gap: 12px; flex-wrap: wrap; }
.rg-btn { display: inline-block; padding: 11px 22px; border-radius: 980px; font-size: 15px; font-weight: 550;
  border: 1px solid var(--border-strong); transition: var(--transition); }
.rg-btn:hover { background: var(--bg-secondary); }
.rg-btn--primary { background: var(--accent); color: #fff; border-color: var(--accent); }
.rg-btn--primary:hover { background: var(--accent-hover); }

.rg-panel { margin-top: 56px; padding-top: 32px; border-top: 1px solid var(--border); }
.rg-panel h2 { font-size: 27px; letter-spacing: -0.018em; font-weight: 640; margin: 0 0 8px; }
.rg-note { font-size: 15px; line-height: 1.66; color: var(--text-secondary); max-width: 700px; margin: 12px 0 22px; }
.rg-note strong { color: var(--text-primary); }

.rg-restable { width: 100%; border-collapse: collapse; font-size: 15px; margin: 18px 0 4px; }
.rg-restable td { padding: 10px 12px; border-bottom: 1px solid var(--border); }
.rg-restable .rg-num { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600;
  font-family: "JetBrains Mono", monospace; font-size: 14px; white-space: nowrap; }
.rg-r--bad .rg-num { color: #c1121f; }
.rg-r--good .rg-num { color: #1a7f37; }
.rg-r--good td:first-child { font-weight: 560; }
.rg-muted { color: var(--text-tertiary); font-size: 13.5px; }

.rg-tier { margin: 26px 0; }
.rg-tier__h { font-size: 14px; font-weight: 640; margin: 0 0 12px; display: flex; gap: 10px;
  align-items: baseline; flex-wrap: wrap; }
.rg-tier__h span { font-weight: 400; color: var(--text-tertiary); font-size: 13.5px; }
.rg-tier__grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px,1fr)); gap: 10px; }
.rg-lvcard { display: flex; align-items: center; gap: 11px; padding: 13px 15px; border-radius: var(--radius-sm);
  border: 1px solid var(--border); background: var(--bg); transition: var(--transition); }
.rg-lvcard:hover { border-color: var(--border-strong); box-shadow: var(--shadow-sm); transform: translateY(-1px); }
.rg-lvcard__n { font-variant-numeric: tabular-nums; font-weight: 650; color: var(--text-tertiary);
  font-size: 13px; min-width: 18px; }
.rg-lvcard__t { flex: 1; font-size: 14px; font-weight: 530; line-height: 1.35; }
.rg-lvcard__s { font-size: 10.5px; letter-spacing: .05em; text-transform: uppercase; font-weight: 600;
  padding: 3px 7px; border-radius: 5px; white-space: nowrap; }
.rg-lvcard--done .rg-lvcard__s { background: rgba(26,127,55,.1); color: #1a7f37; }
.rg-lvcard--wip .rg-lvcard__s { background: var(--bg-secondary); color: var(--text-tertiary); }

.rg-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); gap: 14px; margin-top: 18px; }
.rg-card { display: block; padding: 20px; border: 1px solid var(--border); border-radius: var(--radius-md);
  transition: var(--transition); }
.rg-card:hover { border-color: var(--border-strong); box-shadow: var(--shadow-sm); transform: translateY(-2px); }
.rg-card h4 { font-size: 16px; font-weight: 620; margin: 0 0 7px; }
.rg-card p { font-size: 14px; line-height: 1.6; color: var(--text-secondary); margin: 0; }

.codehilite .k, .codehilite .kn { color: #ff7ab2; }
.codehilite .s, .codehilite .s1, .codehilite .s2 { color: #a5e844; }
.codehilite .c, .codehilite .c1 { color: #6c7986; font-style: italic; }
.codehilite .nf { color: #78c2ff; }
.codehilite .mi, .codehilite .mf { color: #d9c97c; }

@media (max-width: 900px) {
  .rg-doc__wrap { grid-template-columns: 1fr; gap: 24px; }
  .rg-doc__toc { position: static; border-bottom: 1px solid var(--border); padding-bottom: 16px; }
  .rg-doc__toc .rg-lvlist { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px,1fr)); }
  .rg-hero__h { font-size: 36px; }
  .rg-doc__title { font-size: 30px; }
}
"""

if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
