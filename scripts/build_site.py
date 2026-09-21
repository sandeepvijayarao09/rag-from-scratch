"""Generate the /RAG guide for sandeepvijayarao09.github.io.

Emits the same markup the rest of that site uses: dark theme, Geist, absolute
paths, directory routing (/RAG/level-04/), .case-body prose, .case-strip stats,
data-hue tinting, skip link, canonical + OG tags. Only additions are a small
rag.css for the elements the site does not already style (tables, code blocks,
lists, the level grid and the sticky level nav).

House style is enforced here too: the site contains zero em dashes, so the
markdown is normalised on the way through.

    python scripts/build_site.py <output-dir>
"""

import html
import re
import sys
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parents[1]
BASE = "https://sandeepvijayarao09.github.io"
HUE = "llm"
MD = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "toc"],
                       extension_configs={"codehilite": {"noclasses": False,
                                                         "guess_lang": False}})

DONE = {"01", "02", "03", "04", "05", "06", "07", "08", "10", "11", "12"}
TIERS = [("Beginner", "Make it work, then make it measurable", ["01", "02", "03"]),
         ("Core retrieval", "The fundamentals", ["04", "05", "06"]),
         ("Intermediate", "Two-stage and query side", ["07", "08"]),
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


def dedash(text: str) -> str:
    """The site contains no em dashes. Keep it that way."""
    text = re.sub(r"\s+—\s+", ", ", text)
    text = re.sub(r"\s+–\s+", ", ", text)
    return text.replace("—", "-").replace("–", "-")


def shell(title: str, desc: str, url_path: str, body: str) -> str:
    t = html.escape(dedash(title))
    d = html.escape(dedash(desc))
    canon = f"{BASE}{url_path}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{t} | Sandeep Vijayarao</title>
  <meta name="description" content="{d}">
  <link rel="canonical" href="{canon}">
  <meta property="og:title" content="{t} | Sandeep Vijayarao">
  <meta property="og:description" content="{d}">
  <meta property="og:url" content="{canon}">
  <meta property="og:type" content="article">
  <meta name="twitter:card" content="summary">
  <meta name="theme-color" content="#0b0c0f">
  <link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">
  <link rel="preload" href="/assets/fonts/geist-var.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="preload" href="/assets/fonts/geist-mono-var.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="stylesheet" href="/styles.css">
  <link rel="stylesheet" href="/RAG/rag.css">
</head>
<body data-hue="{HUE}">
  <a class="skip" href="#main">Skip to content</a>
  <div id="top-sentinel" aria-hidden="true"></div>

  <header class="nav" id="nav">
    <div class="container nav-inner">
      <a class="nav-name" href="/">Sandeep Vijayarao</a>
      <nav aria-label="Primary">
        <a href="/work/">Work</a>
        <a href="/about/">About</a>
        <a href="/contact/">Contact</a>
      </nav>
    </div>
  </header>

  <main id="main">
{body}
  </main>

  <footer class="footer">
    <div class="container footer-inner">
      <span>&copy; 2026 Sandeep Vijayarao, San Jose, CA</span>
      <span class="footer-links">
        <a href="https://github.com/sandeepvijayarao09" target="_blank" rel="noopener">GitHub</a>
        <a href="https://www.linkedin.com/in/sandeepvijayarao/" target="_blank" rel="noopener">LinkedIn</a>
        <a href="mailto:sandeepvijayarao09@gmail.com">Email</a>
      </span>
    </div>
  </footer>

  <script src="/script.js" defer></script>
</body>
</html>
"""


def render(md_text: str) -> tuple[str, str]:
    MD.reset()
    return MD.convert(dedash(md_text)), getattr(MD, "toc", "")


def split_h1(body: str) -> tuple[str, str]:
    m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    title = re.sub(r"<.*?>", "", m.group(1)).strip() if m else ""
    title = re.sub(r"^Stage\s*\d+\s*[,—–-]\s*", "", title)
    return re.sub(r"<h1[^>]*>.*?</h1>", "", body, count=1, flags=re.S), title


def fix_links(body: str) -> str:
    body = re.sub(r'href="\.\./\.\./([A-Za-z_]+)\.md"', lambda m: f'href="/RAG/{m.group(1).lower()}/"', body)
    body = re.sub(r'href="([A-Z][A-Za-z_]*)\.md"', lambda m: f'href="/RAG/{m.group(1).lower()}/"', body)
    body = re.sub(r'href="(\.\./)?([0-9]{2})_[a-z_]+/"', r'href="/RAG/level-\2/"', body)
    body = re.sub(r'href="(\.\./)*((rag|concepts|scripts)/[^"]+)"',
                  r'href="https://github.com/sandeepvijayarao09/rag-from-scratch/blob/main/\2"', body)
    return body


def level_nav(meta, current) -> str:
    items = "".join(
        f'<li{" class=\"is-here\"" if num == current else ""}>'
        f'<a href="/RAG/level-{num}/"><span>{int(num)}</span>{html.escape(t)}</a></li>'
        for num, t, _ in meta)
    return f'<nav class="rag-levelnav" aria-label="Levels"><p>Levels</p><ul>{items}</ul></nav>'


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "rag.css").write_text(CSS)
    (out / ".nojekyll").write_text("")

    meta = []
    for d in sorted((REPO / "concepts").glob("[0-9][0-9]_*")):
        r = d / "README.md"
        if r.exists():
            body, _ = render(r.read_text())
            body, title = split_h1(body)
            meta.append((d.name[:2], title or d.name, fix_links(body)))

    for i, (num, title, body) in enumerate(meta):
        nxt = ""
        if i + 1 < len(meta):
            n2, t2, _ = meta[i + 1]
            nxt = (f'<a class="next-link" href="/RAG/level-{n2}/" data-reveal><div>'
                   f'<strong>Level {int(n2)}, {html.escape(t2)}</strong><br>'
                   f'<span>Next in the ladder</span></div>'
                   f'<span class="ext" aria-hidden="true">&#8594;</span></a>')
        overview = ('<a class="next-link" href="/RAG/" data-reveal style="--d:.07s"><div>'
                    '<strong>All levels</strong><br><span>The full ladder, fifteen levels '
                    'from naive RAG to production scale</span></div>'
                    '<span class="ext" aria-hidden="true">&#8594;</span></a>')
        page_body = f"""    <header class="page-header container">
      <p class="rag-crumb"><a href="/RAG/">RAG guide</a> / Level {int(num)}</p>
      <h1>{html.escape(title)}</h1>
    </header>

    <div class="container rag-shell">
{level_nav(meta, num)}
      <article class="case-body rag-prose">
{body}
      </article>
    </div>

    <section class="next container">{nxt}{overview}</section>
"""
        d = out / f"level-{num}"
        d.mkdir(exist_ok=True)
        (d / "index.html").write_text(shell(
            title, f"Level {int(num)} of a measured guide to building RAG systems from scratch.",
            f"/RAG/level-{num}/", page_body))

    for src, slug, desc in [("LEARN.md", "learn", "A guided path through building RAG systems, basic to advanced."),
                            ("FLOW.md", "flow", "A decision procedure for building RAG from zero to one."),
                            ("SYSTEMS.md", "systems", "Taxonomy of RAG systems, organised by the problem each solves."),
                            ("NOTES.md", "notes", "Lab notebook. What broke and what I got wrong.")]:
        p = REPO / src
        if not p.exists():
            continue
        body, toc = render(p.read_text())
        body, title = split_h1(body)
        page_body = f"""    <header class="page-header container">
      <p class="rag-crumb"><a href="/RAG/">RAG guide</a> / {html.escape(title)}</p>
      <h1>{html.escape(title)}</h1>
    </header>

    <div class="container rag-shell">
      <nav class="rag-levelnav" aria-label="Contents"><p>On this page</p>{toc}</nav>
      <article class="case-body rag-prose">
{fix_links(body)}
      </article>
    </div>
"""
        d = out / slug
        d.mkdir(exist_ok=True)
        (d / "index.html").write_text(shell(title, desc, f"/RAG/{slug}/", page_body))

    (out / "index.html").write_text(build_index(meta))
    print(f"built {len(meta)} levels + 4 docs + index -> {out}")


def build_index(meta) -> str:
    titles = {n: t for n, t, _ in meta}
    tiers = ""
    for name, sub, nums in TIERS:
        cards = "".join(
            f'<a class="rag-lv{" is-wip" if n not in DONE else ""}" href="/RAG/level-{n}/" data-reveal>'
            f'<span class="rag-lv-n">{int(n)}</span>'
            f'<span class="rag-lv-t">{html.escape(titles[n])}</span>'
            f'<span class="rag-lv-s">{"measured" if n in DONE else "in progress"}</span></a>'
            for n in nums if n in titles)
        tiers += (f'<div class="rag-tier"><h3>{name}<span>{sub}</span></h3>'
                  f'<div class="rag-tier-grid">{cards}</div></div>')

    res = "".join(
        f'<tr class="rag-{k}"><td>{html.escape(t)}'
        + (f' <em>{html.escape(s)}</em>' if s else '') + f'</td><td>{d}</td></tr>'
        for t, s, d, k in RESULTS)

    body = f"""    <header class="page-header container">
      <h1>Building RAG from scratch</h1>
      <p class="lede">I implemented ten recommended RAG techniques and measured each one
      against the same benchmark. Eight made retrieval worse. This is the code, the numbers,
      and the reasoning for why.</p>
    </header>

    <section class="case-strip">
      <div class="container case-strip-grid">
        <div class="stat"><span class="stat-num">5,183</span><span class="stat-label">abstracts, BEIR SciFact</span></div>
        <div class="stat"><span class="stat-num">300</span><span class="stat-label">labelled queries</span></div>
        <div class="stat"><span class="stat-num">15</span><span class="stat-label">levels, naive to production</span></div>
      </div>
    </section>

    <article class="case-body rag-prose container">
      <div class="case-actions">
        <a class="btn btn-primary" href="/RAG/learn/">Start the guide</a>
        <a class="btn btn-ghost" href="https://github.com/sandeepvijayarao09/rag-from-scratch" target="_blank" rel="noopener">View the code</a>
      </div>

      <h2>What I measured</h2>
      <table class="rag-results"><tbody>{res}</tbody></table>
      <p>The two clear wins are the two where I measured the gap before applying the technique.
      None of this means these techniques are bad. It means they are conditional, and the
      condition is almost always whether your pipeline is already good at the thing the
      technique fixes. Reranking flipped sign inside this repo: -0.033 on a strong first stage,
      +0.041 on a weak one.</p>
      <p class="case-note">Baseline validated against a published bge-base-en-v1.5 score of
      roughly 0.741 before anything was built on top of it. At n=300, deltas under about 0.02
      sit inside the noise floor, and the write-ups say so.</p>

      <h2>The levels</h2>
      <p>Each level is a distinct kind of RAG system, ordered so every level only needs what
      came before it. Code and a written finding for each.</p>
      {tiers}

      <h2>Read it a different way</h2>
      <div class="rag-cards">
        <a class="rag-card" href="/RAG/learn/"><strong>Learner's guide</strong>
          <span>Eight parts, basic to advanced. Explains each mechanism before showing the
          result, with something to run at every step.</span></a>
        <a class="rag-card" href="/RAG/flow/"><strong>Decision procedure</strong>
          <span>The same material as a build order, with a diagnostic gate before each
          technique. For when you already know the techniques.</span></a>
        <a class="rag-card" href="/RAG/systems/"><strong>Taxonomy</strong>
          <span>Every kind of RAG system, organised by the problem it solves rather than by
          how fashionable it is.</span></a>
        <a class="rag-card" href="/RAG/notes/"><strong>Lab notebook</strong>
          <span>What broke, what I got wrong, and the open questions. Including two
          hypotheses I falsified before landing the reranking rule.</span></a>
      </div>
    </article>
"""
    return shell("Building RAG from scratch",
                 "A measured, level by level guide to building RAG systems. Ten techniques "
                 "tested on BEIR SciFact; eight made retrieval worse.", "/RAG/", body)


CSS = """/* /RAG guide. Extends styles.css; adds only what the site does not already style. */
.rag-shell { display: grid; grid-template-columns: 220px minmax(0,1fr); gap: 3rem;
  align-items: start; padding-bottom: 4rem; }
.rag-prose { max-width: none; }
.rag-crumb { font-family: var(--mono); font-size: .78rem; color: var(--muted);
  letter-spacing: .02em; margin-bottom: .6rem; }
.rag-crumb a { color: var(--accent); text-decoration: none; }

.rag-levelnav { position: sticky; top: 5rem; font-size: .84rem; }
.rag-levelnav > p { font-family: var(--mono); font-size: .7rem; text-transform: uppercase;
  letter-spacing: .12em; color: var(--muted); margin-bottom: .7rem; }
.rag-levelnav ul { list-style: none; }
.rag-levelnav a { display: flex; gap: .55rem; padding: .32rem .5rem; border-radius: var(--r-token);
  color: var(--muted); text-decoration: none; line-height: 1.35; transition: .18s var(--ease); }
.rag-levelnav a:hover { color: var(--text); background: var(--surface); }
.rag-levelnav a span { font-family: var(--mono); font-size: .72rem; opacity: .6; min-width: 1.1rem; }
.rag-levelnav .is-here > a { color: var(--accent); background: var(--accent-dim); }
.rag-levelnav > ul ul { margin-left: .7rem; border-left: 1px solid var(--line); padding-left: .4rem; }

.rag-prose h2 { margin-top: 2.6rem; }
.rag-prose h3 { margin-top: 1.9rem; font-size: 1.05rem; }
.rag-prose ul, .rag-prose ol { margin: 0 0 1.1rem 1.15rem; color: var(--muted); }
.rag-prose li { margin-bottom: .4rem; }
.rag-prose li > strong { color: var(--text); }
.rag-prose p a, .rag-prose li a, .rag-prose td a,
.rag-prose blockquote a { color: var(--accent); }
.rag-prose .btn-primary { color: var(--accent-ink); }
.rag-prose strong { color: var(--text); }
.rag-prose blockquote { margin: 1.4rem 0; padding: .9rem 1.2rem; border-left: 2px solid var(--accent);
  background: var(--bg-2); border-radius: 0 var(--r-token) var(--r-token) 0; color: var(--muted); }
.rag-prose blockquote p { margin: 0; }
.rag-prose blockquote p + p { margin-top: .7rem; }

.rag-prose pre { background: var(--bg-2); border: 1px solid var(--line); border-radius: var(--r-token);
  padding: 1rem 1.1rem; overflow-x: auto; margin: 0 0 1.3rem; font-family: var(--mono);
  font-size: .78rem; line-height: 1.6; color: var(--text); }
.rag-prose code { font-family: var(--mono); font-size: .84em; background: var(--surface);
  padding: .12em .38em; border-radius: 4px; color: var(--text); }
.rag-prose pre code { background: none; padding: 0; font-size: inherit; }

.rag-prose table { width: 100%; border-collapse: collapse; margin: 0 0 1.4rem; font-size: .86rem; }
.rag-prose th { text-align: left; font-family: var(--mono); font-size: .68rem; font-weight: 500;
  letter-spacing: .1em; text-transform: uppercase; color: var(--muted);
  padding: .5rem .7rem; border-bottom: 1px solid var(--line); }
.rag-prose td { padding: .55rem .7rem; border-bottom: 1px solid var(--line);
  color: var(--muted); vertical-align: top; }
.rag-prose td strong { color: var(--text); }

.rag-results { width: 100%; border-collapse: collapse; font-size: .92rem; margin: 1.2rem 0 1.6rem; }
.rag-results td { padding: .6rem .7rem; border-bottom: 1px solid var(--line); color: var(--text); }
.rag-results td:last-child { text-align: right; font-family: var(--mono); font-size: .84rem;
  white-space: nowrap; }
.rag-results em { color: var(--muted); font-style: normal; font-size: .82rem; }
.rag-bad td:last-child { color: #fb7185; }
.rag-good td:last-child { color: var(--accent); }
.rag-good td:first-child { font-weight: 500; }

.rag-tier { margin: 1.6rem 0; }
.rag-tier h3 { font-size: .8rem; font-family: var(--mono); text-transform: uppercase;
  letter-spacing: .1em; color: var(--text); margin: 0 0 .7rem; display: flex; gap: .7rem;
  flex-wrap: wrap; align-items: baseline; }
.rag-tier h3 span { font-family: var(--sans); text-transform: none; letter-spacing: 0;
  color: var(--muted); font-size: .84rem; }
.rag-tier-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px,1fr)); gap: .6rem; }
.rag-lv { display: flex; align-items: center; gap: .6rem; padding: .7rem .85rem;
  border: 1px solid var(--line); border-radius: var(--r-token); background: var(--surface);
  text-decoration: none; transition: .18s var(--ease); }
.rag-lv:hover { border-color: rgba(var(--accent-rgb), .5); transform: translateY(-1px); }
.rag-lv-n { font-family: var(--mono); font-size: .74rem; color: var(--muted); min-width: 1.1rem; }
.rag-lv-t { flex: 1; font-size: .87rem; color: var(--text); line-height: 1.3; }
.rag-lv-s { font-family: var(--mono); font-size: .6rem; text-transform: uppercase;
  letter-spacing: .08em; padding: .18rem .4rem; border-radius: 4px;
  background: var(--accent-dim); color: var(--accent); white-space: nowrap; }
.rag-lv.is-wip .rag-lv-s { background: var(--bg-2); color: var(--muted); }

.rag-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px,1fr));
  gap: .8rem; margin-top: 1.1rem; }
.rag-card { display: block; padding: 1.1rem; border: 1px solid var(--line);
  border-radius: var(--r-surface); background: var(--surface); text-decoration: none;
  transition: .18s var(--ease); }
.rag-card:hover { border-color: rgba(var(--accent-rgb), .5); transform: translateY(-2px); }
.rag-card strong { display: block; color: var(--text); margin-bottom: .35rem; font-size: .95rem; }
.rag-card span { color: var(--muted); font-size: .84rem; line-height: 1.55; }

.codehilite .k, .codehilite .kn { color: #a78bfa; }
.codehilite .s, .codehilite .s1, .codehilite .s2 { color: var(--accent); }
.codehilite .c, .codehilite .c1 { color: #5b6470; font-style: italic; }
.codehilite .nf { color: #38bdf8; }
.codehilite .mi, .codehilite .mf { color: #fbbf24; }

@media (max-width: 860px) {
  .rag-shell { grid-template-columns: 1fr; gap: 1.4rem; }
  .rag-levelnav { position: static; border-bottom: 1px solid var(--line); padding-bottom: 1rem; }
  .rag-levelnav ul { display: grid; grid-template-columns: repeat(auto-fill, minmax(165px,1fr)); }
}
"""

if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
