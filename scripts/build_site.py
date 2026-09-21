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
HUE = "agent"
MD = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "toc"],
                       extension_configs={"codehilite": {"noclasses": False,
                                                         "guess_lang": False}})

DONE = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"}
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
           ("Contextual retrieval", "on self-contained chunks", "-0.006", "bad"),
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

    levels = "".join(
        f'<a class="rag-row{"" if n in DONE else " is-wip"}" href="/RAG/level-{n}/">'
        f'<span class="rag-row-n">{int(n)}</span>'
        f'<span class="rag-row-t">{html.escape(titles[n])}</span></a>'
        for n, _, _ in meta)

    lost = [(t, d) for t, _, d, k in RESULTS if k == "bad"]
    won = [(t, d) for t, _, d, k in RESULTS if k == "good"]
    lost_rows = "".join(f'<li><span>{html.escape(t)}</span><em>{d}</em></li>' for t, d in lost)
    won_rows = "".join(f'<li><span>{html.escape(t)}</span><em>{d}</em></li>' for t, d in won)

    body = f"""    <header class="page-header container">
      <h1>Building RAG from scratch</h1>
      <p class="lede">I built fifteen kinds of RAG system and measured each one on the same
      benchmark. Eight of the ten techniques I tested made retrieval worse.</p>
      <div class="case-actions">
        <a class="btn btn-primary" href="/RAG/learn/">Read the guide</a>
        <a class="btn btn-ghost" href="https://github.com/sandeepvijayarao09/rag-from-scratch" target="_blank" rel="noopener">Code</a>
      </div>
    </header>

    <article class="case-body rag-prose container">
      <div class="rag-split">
        <div>
          <h3 class="rag-h">Made it worse</h3>
          <ul class="rag-scores rag-lost">{lost_rows}</ul>
        </div>
        <div>
          <h3 class="rag-h">Helped</h3>
          <ul class="rag-scores rag-won">{won_rows}</ul>
        </div>
      </div>

      <p>Change in nDCG@10 against a 0.759 baseline on BEIR SciFact, 5,183 abstracts and
      300 labelled queries. The two that helped are the two where I measured the gap before
      applying the technique. None of these are bad techniques. They are conditional, and
      the condition is whether your pipeline is already good at the thing they fix.
      Reranking flipped sign inside this repo: -0.033 on a strong first stage, +0.041 on a
      weak one.</p>

      <h2>The fifteen levels</h2>
      <div class="rag-rows">{levels}</div>

      <h2>More</h2>
      <p><a href="/RAG/flow/">Decision procedure</a>, a build order with a diagnostic before
      each technique. <a href="/RAG/systems/">Taxonomy</a> of every RAG system by the problem
      it solves. <a href="/RAG/notes/">Lab notebook</a>, what broke and what I got wrong.</p>
    </article>
"""
    return shell("Building RAG from scratch",
                 "I built fifteen kinds of RAG system and measured each one. Eight of ten "
                 "techniques made retrieval worse.", "/RAG/", body)


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


.rag-split { display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem; margin: .5rem 0 1.6rem; }
.rag-h { font-family: var(--mono); font-size: .68rem; text-transform: uppercase;
  letter-spacing: .12em; color: var(--muted); margin: 0 0 .7rem; font-weight: 500; }
.rag-scores { list-style: none; margin: 0; }
.rag-scores li { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline;
  padding: .42rem 0; border-bottom: 1px solid var(--line); font-size: .9rem; }
.rag-scores li:last-child { border-bottom: none; }
.rag-scores span { color: var(--text); }
.rag-scores em { font-family: var(--mono); font-size: .8rem; font-style: normal; white-space: nowrap; }
.rag-lost em { color: #fb7185; }
.rag-won em { color: var(--accent); }
.rag-won span { font-weight: 500; }

.rag-rows { display: grid; grid-template-columns: 1fr 1fr; gap: 0 2.5rem; margin: 1rem 0 1.6rem; }
.rag-row { display: flex; gap: .8rem; align-items: baseline; padding: .5rem 0;
  border-bottom: 1px solid var(--line); text-decoration: none; transition: .15s var(--ease); }
.rag-row:hover { padding-left: .35rem; }
.rag-row:hover .rag-row-t { color: var(--accent); }
.rag-row-n { font-family: var(--mono); font-size: .75rem; color: var(--muted); min-width: 1.4rem; }
.rag-row-t { color: var(--text); font-size: .92rem; }
.rag-row.is-wip .rag-row-t { color: var(--muted); }
.rag-row.is-wip .rag-row-t::after { content: " in progress"; font-family: var(--mono);
  font-size: .62rem; text-transform: uppercase; letter-spacing: .08em; opacity: .55; }

.codehilite .k, .codehilite .kn { color: #a78bfa; }
.codehilite .nc, .codehilite .nn { color: var(--accent); }
.codehilite .s, .codehilite .s1, .codehilite .s2 { color: var(--accent); }
.codehilite .c, .codehilite .c1 { color: #5b6470; font-style: italic; }
.codehilite .nf { color: #38bdf8; }
.codehilite .mi, .codehilite .mf { color: #fbbf24; }

@media (max-width: 720px) {
  .rag-split, .rag-rows { grid-template-columns: 1fr; gap: 1.6rem 0; }
}

@media (max-width: 860px) {
  .rag-shell { grid-template-columns: 1fr; gap: 1.4rem; }
  .rag-levelnav { position: static; border-bottom: 1px solid var(--line); padding-bottom: 1rem; }
  .rag-levelnav ul { display: grid; grid-template-columns: repeat(auto-fill, minmax(165px,1fr)); }
}
"""

if __name__ == "__main__":
    build(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
