"""Build the Quarto sources of the documentation website from the repository.

The Markdown files, notebooks and docstrings on GitHub are the only sources.
This script copies them into site/_build/, with
  * README.md as the home page, the README of each pipeline step and of the
    Sentinel-2 demo as one page each, every docs/*.md as one page, and
    CONTRIBUTING.md under "Project";
  * links rewritten: links between site pages point to the site, and links to
    any other file of the repository point to that file on GitHub;
  * at the top of each page, a link back to the same file on GitHub (and, for
    a notebook, a link to open it in Colab);
  * a generated _quarto.yml, including the quartodoc configuration of the API
    reference, which `quartodoc build` writes from the docstrings of the step
    scripts into api/ (read statically; nothing is imported or run).
New files in docs/ appear on the site and in its navbar without further
changes.

Usage (from the repository root, with quartodoc installed):
    python3 site/build_site.py
    cd site/_build
    PYTHONPATH=../../step2_mosaic:../../step3_pan_mosaic:../../step4_cloud_removal python -m quartodoc build
    cd ../..
    quarto render site/_build          # output in site/_build/_site/
"""
import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "site" / "_build"
GITHUB = "https://github.com/buckai-observatory/Automated-Cloud-Removal-and-Seamless-Satellite-Image-Mosaicking-Pipeline"
SITE_URL = "https://buckai-observatory.org/Automated-Cloud-Removal-and-Seamless-Satellite-Image-Mosaicking-Pipeline/"
BRANCH = "main"

# source file (relative to the repository root) -> (page file in the site, navbar name)
STEPS = {
    "step1_color_balance/README.md": ("step1_color_balance.md", "Step 1: color balance"),
    "step2_mosaic/README.md": ("step2_mosaic.md", "Step 2: seamline mosaic"),
    "step3_pan_mosaic/README.md": ("step3_pan_mosaic.md", "Step 3: Pan mosaic"),
    "step4_cloud_removal/README.md": ("step4_cloud_removal.md", "Step 4: cloud filling"),
}
DEMO = {"examples/sentinel2_demo/README.md": ("sentinel2_demo.md", "Sentinel-2 demo")}
PROJECT = {"CONTRIBUTING.md": "Contributing"}   # project files -> navbar names
PAGES = {"README.md": "index.md"}
for source, (target, _) in {**STEPS, **DEMO}.items():
    PAGES[source] = target
for f in sorted((REPO / "docs").glob("*.md")):
    PAGES[f"docs/{f.name}"] = f.name
for name in PROJECT:
    if (REPO / name).exists():
        PAGES[name] = name
ASSETS = []                                            # copied into the site as they are

# API reference: one section per module, listing its public functions and
# classes. quartodoc renders them from the docstrings.
API_SECTIONS = [
    ("Step 2: seamline mosaic", "`step2_mosaic/step2mosaic.py`", "step2mosaic",
     ["main", "polygonize_valid_area", "test_overlap", "find_raster_intersect",
      "find_raster_intersect_for_mosaic", "clipImg", "MakeTrueID", "buildGraph",
      "border_connection", "bfs", "spiral", "add_at_pos", "blendMask",
      "blend_preprocess", "blend_preprocess_for_order", "blend_preprocess_for_mosaic",
      "blend_preprocess_order_for_mosaic", "mask_blend_preprocess_for_order",
      "mask_blend_preprocess_for_order_loop"]),
    ("Step 3: Pan mosaic", "`step3_pan_mosaic/mosaicPan_block_rs.py`", "mosaicPan_block_rs",
     ["main", "pan_scene_path"]),
    ("Step 4: cloud filling", "`step4_cloud_removal/step4cloud_removal.py`", "step4cloud_removal",
     ["main", "patch_margins", "seamless_clone_over_nodata", "border_elems_generic",
      "pan_scene_path"]),
]

LINK = re.compile(r"(\]\()([^)\s]+)(\))")
H1 = re.compile(r"^# (.+)$", re.M)


def site_name(target: str) -> str:
    return re.sub(r"\.(md|ipynb)$", ".html", target)


def resolve(target: str, source: str) -> str:
    """Rewrite one link target of the file `source` (both relative to the repository)."""
    if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
        return target
    path, _, anchor = target.partition("#")
    repo_path = (REPO / Path(source).parent / path).resolve().relative_to(REPO).as_posix()
    here = Path(PAGES[source]).parent
    if repo_path in PAGES:
        url = os.path.relpath(site_name(PAGES[repo_path]), here)
    elif any(repo_path == a or repo_path.startswith(a + "/") for a in ASSETS):
        url = os.path.relpath(repo_path, here)
    elif (REPO / repo_path).is_dir():
        url = f"{GITHUB}/tree/{BRANCH}/{repo_path}".rstrip("/.")
    else:
        url = f"{GITHUB}/blob/{BRANCH}/{repo_path}"
    return url + (f"#{anchor}" if anchor else "")


def is_notebook(url: str) -> bool:
    return (url.split("#")[0].endswith(".ipynb") or "colab.research.google.com/github" in url
            or re.search(r"(^|/)notebooks/[^/#]+\.html", url) is not None)


NEW_TAB = '{target="_blank"}'


def relink(text: str, source: str) -> str:
    """Rewrite the links of a page; links to notebooks open a new tab."""
    def one(m):
        url = resolve(m.group(2), source)
        tail = text[m.end():m.end() + 1]
        attr = NEW_TAB if is_notebook(url) and tail != "{" else ""
        return m.group(1) + url + m.group(3) + attr
    return LINK.sub(one, text)


def md_page(source: str) -> tuple[str, str]:
    text = (REPO / source).read_text()
    # the home page does not link to itself
    text = "\n".join(l for l in text.splitlines() if SITE_URL not in l) + "\n"
    m = H1.search(text)
    title = m.group(1).strip() if m else Path(source).stem
    if m:
        text = text[:m.start()] + text[m.end():].lstrip("\n")
    back = (f"[View this page on GitHub]({GITHUB}/blob/{BRANCH}/{source})"
            "{.github-link}\n\n")
    return title, f'---\ntitle: "{title}"\n---\n\n' + back + relink(text, source)


def quartodoc_yaml() -> str:
    lines = ["quartodoc:",
             "  package: null                  # objects below are given by their full paths",
             "  dir: api",
             "  title: \"API reference\"",
             "  sidebar: api/_sidebar.yml",
             "  sections:"]
    for title, desc, module, names in API_SECTIONS:
        lines += [f"    - title: \"{title}\"", f"      desc: \"{desc}\"", "      contents:"]
        lines += [f"        - {module}.{n}" for n in names]
    return "\n".join(lines) + "\n"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    docs, steps, project = [], [], []
    for source, target in PAGES.items():
        dst = OUT / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        title, body = md_page(source)
        dst.write_text(body)
        if source in PROJECT:
            project.append((target, PROJECT[source]))
        elif source in STEPS or source in DEMO:
            steps.append((target, {**STEPS, **DEMO}[source][1]))
        else:
            docs.append((target, "Home" if target == "index.md" else title))
    for a in ASSETS:
        if (REPO / a).exists():
            shutil.copytree(REPO / a, OUT / a)
    shutil.copy(REPO / "site" / "styles.css", OUT / "styles.css")
    # quartodoc writes the sidebar on `quartodoc build`; an empty one lets
    # `quarto render` run on its own as well
    (OUT / "api").mkdir()
    (OUT / "api" / "_sidebar.yml").write_text("website:\n  sidebar: []\n")
    doc_items = "\n".join(f"      - href: {t}\n        text: \"{n}\"" for t, n in docs)
    step_items = "\n".join(f"          - href: {t}\n            text: \"{n}\"" for t, n in steps)
    project_items = "\n".join(f"          - href: {t}\n            text: \"{n}\"" for t, n in project)
    (OUT / "_quarto.yml").write_text(f"""project:
  type: website
  output-dir: _site

metadata-files:
  - api/_sidebar.yml

website:
  title: "Cloud removal and seamless mosaicking"
  site-url: {SITE_URL}
  repo-url: {GITHUB}
  repo-actions: [issue]
  navbar:
    left:
{doc_items}
      - text: "Pipeline"
        menu:
{step_items}
      - href: api/index.qmd
        text: "API reference"
      - text: "Project"
        menu:
{project_items}
    right:
      - icon: github
        href: {GITHUB}
        aria-label: GitHub
  page-footer:
    center: "Generated from the Markdown files, notebooks and docstrings of the [repository]({GITHUB}); edit those, not this site."

format:
  html:
    theme:
      light: cosmo
      dark: darkly
    css: styles.css
    toc: true

""" + quartodoc_yaml())
    print(f"wrote {len(docs) + len(steps) + len(project)} pages to {OUT}")


if __name__ == "__main__":
    main()
