#!/usr/bin/env python3
"""Digitize matplotlib-generated SVG figures back into numeric CSV data.

Method:
  - Parse SVG XML (with comments preserved; matplotlib embeds the rendered
    text of every text group as an XML comment, glyph <use> refs are the
    fallback decode path).
  - For each <g id="axes_N">: read tick-mark <use> positions inside
    xtick_*/ytick_* groups, decode tick labels, fit linear or log10
    pixel->data mappings.
  - Extract every data <g id="line2d_*"> / <g id="PathCollection_*">
    path/marker vertices, map to data units, save one CSV per curve.
  - Decode legends to map stroke colors -> labels (e.g. "218 K").

Only stdlib + numpy.
"""

import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

SVG_NS = "{http://www.w3.org/2000/svg}"
XLINK_NS = "{http://www.w3.org/1999/xlink}"

import os
IN_DIR = os.environ.get("GANPLL_FIG_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ""))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FIGS = ["F2_", "F9_", "F11_", "F12_", "F14_"]

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# ----------------------------------------------------------------------
# parsing helpers
# ----------------------------------------------------------------------

def parse_svg(path):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(path, parser=parser)
    return tree.getroot()


def tag_name(el):
    if callable(el.tag):  # comment / PI nodes
        return "!comment"
    return el.tag.replace(SVG_NS, "")


def iter_with_parents(root):
    """Yield (element, parent_chain_ids) for every element."""
    stack = [(root, [])]
    while stack:
        el, chain = stack.pop()
        yield el, chain
        eid = el.get("id") if not callable(el.tag) else None
        child_chain = chain + ([eid] if eid else [])
        for child in reversed(list(el)):
            stack.append((child, child_chain))


def iter_no_defs(root):
    """Like iter_with_parents but does not descend into <defs>."""
    stack = [root]
    while stack:
        el = stack.pop()
        yield el
        if callable(el.tag) or tag_name(el) == "defs":
            continue
        stack.extend(reversed(list(el)))


def find_groups(root, prefix):
    """All <g> whose id starts with prefix, with parent id chains."""
    out = []
    for el, chain in iter_with_parents(root):
        if not callable(el.tag) and tag_name(el) == "g":
            eid = el.get("id", "")
            if eid.startswith(prefix):
                out.append((el, chain))
    return out


def path_vertices(d):
    """Extract vertex list from an SVG path 'd' (absolute M/L/C/Q/z as
    matplotlib emits). For curves (C/Q) keep the on-curve endpoint."""
    verts = []
    i = 0
    n = len(d)
    cmd = None
    nums = FLOAT_RE
    pos = 0
    tokens = re.findall(r"[MLCQZzmlcq]|" + FLOAT_RE.pattern, d)
    ti = 0
    while ti < len(tokens):
        t = tokens[ti]
        if t.isalpha():
            cmd = t
            ti += 1
            continue
        if cmd in ("M", "L"):
            verts.append((float(tokens[ti]), float(tokens[ti + 1])))
            ti += 2
        elif cmd == "C":
            # 3 coordinate pairs; endpoint is on-curve
            verts.append((float(tokens[ti + 4]), float(tokens[ti + 5])))
            ti += 6
        elif cmd == "Q":
            verts.append((float(tokens[ti + 2]), float(tokens[ti + 3])))
            ti += 4
        else:  # lowercase/relative unexpected from mpl; bail on this token
            ti += 1
    return verts


def style_get(el, key):
    st = el.get("style", "") or ""
    m = re.search(key + r":\s*([^;]+)", st)
    return m.group(1).strip() if m else None


# ----------------------------------------------------------------------
# text decoding
# ----------------------------------------------------------------------

def decode_glyph_run(g):
    """Fallback: decode <use xlink:href="#DejaVuSerif-XX"> runs to text."""
    chars = []
    for el, _ in iter_with_parents(g):
        if callable(el.tag) or tag_name(el) != "use":
            continue
        href = el.get(XLINK_NS + "href", "")
        m = re.match(r"#DejaVu\w*-([0-9a-fA-F]+)$", href)
        if m:
            try:
                chars.append(chr(int(m.group(1), 16)))
            except ValueError:
                pass  # skip undecodable glyph
    return "".join(chars)


def text_of_group(g):
    """Decoded text of a matplotlib text <g>: prefer embedded comment."""
    for el, _ in iter_with_parents(g):
        if callable(el.tag):  # comment
            txt = (el.text or "").strip()
            if txt:
                return txt
    t = decode_glyph_run(g)
    return t if t else None


def parse_tick_value(label):
    """Tick label string -> float, handling minus sign and 10^{k} math."""
    if label is None:
        return None
    s = label.replace("−", "-").replace("×", "x").strip()
    # $\mathdefault{10^{6}}$  or  $\mathdefault{2\times10^{6}}$
    m = re.match(
        r"^\$?\\?mathdefault\{?\s*(?:([-\d.]+)\s*\\times\s*)?10\^\{?(-?\d+)\}?\}?\$?$",
        s.replace("$", "").replace("\\mathdefault", "mathdefault"),
    )
    if "mathdefault" in s or "10^" in s:
        m2 = re.search(r"(?:([-\d.]+)\s*(?:\\times|x)\s*)?10\^\{?(-?\d+)\}?", s)
        if m2:
            mant = float(m2.group(1)) if m2.group(1) else 1.0
            return mant * 10.0 ** int(m2.group(2))
    try:
        return float(s)
    except ValueError:
        m3 = FLOAT_RE.search(s)
        return float(m3.group(0)) if m3 else None


def is_pow10_label(label):
    return label is not None and ("10^" in label.replace(" ", ""))


def parse_offset_text(label):
    """Axis offset/exponent text like '1e-9' or '×10^-9' -> multiplier."""
    if label is None:
        return None
    s = label.replace("−", "-").replace("$", "")
    m = re.search(r"(?:\\times|×|x)?\s*10\^\{?(-?\d+)\}?", s)
    if m and "10" in s:
        return 10.0 ** int(m.group(1))
    m = re.match(r"^\s*1e(-?\d+)\s*$", s)
    if m:
        return 10.0 ** int(m.group(1))
    return None


# ----------------------------------------------------------------------
# axes model
# ----------------------------------------------------------------------

class AxisMap:
    def __init__(self, ticks, which):
        """ticks: list of (pixel, value, label). which: 'x' or 'y'."""
        self.which = which
        self.ticks = ticks
        self.scale = None
        self.ok = False
        self.slope = self.intercept = None
        labeled = [(p, v) for p, v, lab in ticks if v is not None]
        # dedupe pixels
        seen = {}
        for p, v in labeled:
            seen[round(p, 3)] = v
        labeled = sorted(seen.items())
        if len(labeled) < 2:
            return
        px = np.array([p for p, _ in labeled])
        vals = np.array([v for _, v in labeled])
        loggy = all(is_pow10_label(lab) for _, v, lab in ticks
                    if v is not None) and np.all(vals > 0)
        if not loggy and np.all(vals > 0) and len(vals) >= 3:
            # compare linear vs log fit quality
            lin_r = self._fit_resid(px, vals)
            log_r = self._fit_resid(px, np.log10(vals))
            # ratio-spaced values fit log10 far better
            loggy = log_r < lin_r * 1e-3 and not np.allclose(
                np.diff(vals), np.diff(vals)[0], rtol=1e-6)
        y = np.log10(vals) if loggy else vals
        A = np.vstack([px, np.ones_like(px)]).T
        (m, b), *_ = np.linalg.lstsq(A, y, rcond=None)
        # categorical axis check: evenly spaced pixel ticks whose values
        # do not fit an affine map (e.g. bar chart '218K','273K','300K')
        resid = self._fit_resid(px, y)
        if not loggy and len(vals) >= 3 and resid > 0.02:
            self.scale = "categorical"
            self.cats = list(zip(px.tolist(), vals.tolist()))
            self.ok = True
            return
        self.slope, self.intercept = float(m), float(b)
        self.scale = "log" if loggy else "linear"
        self.ok = True

    @staticmethod
    def _fit_resid(x, y):
        A = np.vstack([x, np.ones_like(x)]).T
        sol, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ sol
        return float(np.sqrt(np.mean((pred - y) ** 2)) /
                     (np.std(y) + 1e-30))

    def to_data(self, pix):
        pix = np.asarray(pix, dtype=float)
        if self.scale == "categorical":
            cpx = np.array([p for p, _ in self.cats])
            cv = np.array([v for _, v in self.cats])
            idx = np.abs(pix[:, None] - cpx[None, :]).argmin(axis=1)
            return cv[idx]
        v = self.slope * pix + self.intercept
        return 10.0 ** v if self.scale == "log" else v


def collect_ticks(axes_g, kind):
    """kind in ('xtick', 'ytick') -> list of (pixel, value, label)."""
    ticks = []
    for el, chain in iter_with_parents(axes_g):
        if callable(el.tag) or tag_name(el) != "g":
            continue
        eid = el.get("id", "")
        if not eid.startswith(kind):
            continue
        # tick mark position: first <use> of tick def
        pix = None
        for sub, _ in iter_with_parents(el):
            if not callable(sub.tag) and tag_name(sub) == "use":
                href = sub.get(XLINK_NS + "href", "")
                if not href.startswith("#DejaVu"):
                    pix = float(sub.get("x" if kind == "xtick" else "y"))
                    break
        if pix is None:
            continue
        # label: text_* subgroup
        label = None
        for sub in el:
            if not callable(sub.tag) and tag_name(sub) == "g" and \
                    sub.get("id", "").startswith("text_"):
                label = text_of_group(sub)
        ticks.append((pix, parse_tick_value(label), label))
    return ticks


def axes_bbox(axes_g):
    """Bounding rect of the axes background patch (first patch_ path)."""
    for el in axes_g:
        if callable(el.tag):
            continue
        if tag_name(el) == "g" and el.get("id", "").startswith("patch_"):
            for sub, _ in iter_with_parents(el):
                if not callable(sub.tag) and tag_name(sub) == "path":
                    v = path_vertices(sub.get("d", ""))
                    if len(v) >= 4:
                        xs = [p[0] for p in v]
                        ys = [p[1] for p in v]
                        return (min(xs), min(ys), max(xs), max(ys))
    return None


# ----------------------------------------------------------------------
# curves & legends
# ----------------------------------------------------------------------

def extract_curves(axes_g):
    """line2d_*/PathCollection_* data inside axes, excluding axis
    machinery and legends. Returns list of dicts."""
    curves = []
    for el, chain in iter_with_parents(axes_g):
        if callable(el.tag) or tag_name(el) != "g":
            continue
        eid = el.get("id", "")
        if not (eid.startswith("line2d_") or eid.startswith("PathCollection_")):
            continue
        if any(c and (c.startswith("matplotlib.axis") or
                      c.startswith("xtick") or c.startswith("ytick") or
                      c.startswith("legend")) for c in chain):
            continue
        pts = []
        color = None
        kind = "line"
        # stroked path with vertices (skip <defs>: marker outline shapes)
        for sub in iter_no_defs(el):
            if callable(sub.tag):
                continue
            tn = tag_name(sub)
            if tn == "path" and sub.get("d"):
                stroke = style_get(sub, "stroke")
                v = path_vertices(sub.get("d"))
                if len(v) >= 2 and stroke and stroke != "none":
                    pts.extend(v)
                    color = color or stroke
                    dash = style_get(sub, "stroke-dasharray")
                    if dash:
                        kind = "dashed-line"
        if not pts:
            # marker-based (defs + use)
            fill = None
            for sub in iter_no_defs(el):
                if callable(sub.tag):
                    continue
                tn = tag_name(sub)
                if tn == "use":
                    href = sub.get(XLINK_NS + "href", "")
                    if href.startswith("#DejaVu"):
                        continue
                    x, y = sub.get("x"), sub.get("y")
                    if x is not None and y is not None:
                        pts.append((float(x), float(y)))
                    if fill is None:
                        f = style_get(sub, "fill") or style_get(sub, "stroke")
                        if f and f != "none":
                            fill = f
            color = fill
            kind = "markers"
        if len(pts) >= 2:
            curves.append({"id": eid, "color": (color or "").lower(),
                           "kind": kind, "pts": pts})
    curves.extend(extract_bars(axes_g))
    return curves


def extract_bars(axes_g):
    """Bar-chart rectangles: <g id="patch_*"> with colored fill inside the
    axes. Grouped by fill color into one series per color; each bar
    becomes a point (x_center, y_top) in pixel space."""
    by_color = {}
    for el, chain in iter_with_parents(axes_g):
        if callable(el.tag) or tag_name(el) != "g":
            continue
        if not el.get("id", "").startswith("patch_"):
            continue
        if any(c and (c.startswith("matplotlib.axis") or
                      c.startswith("legend") or c.startswith("xtick") or
                      c.startswith("ytick")) for c in chain):
            continue
        for sub, _ in iter_with_parents(el):
            if callable(sub.tag) or tag_name(sub) != "path":
                continue
            fill = style_get(sub, "fill")
            if not fill or fill in ("none", "#ffffff"):
                continue
            v = path_vertices(sub.get("d", ""))
            if not (4 <= len(v) <= 5):
                continue
            xs = sorted(set(round(p[0], 4) for p in v))
            ys = sorted(set(round(p[1], 4) for p in v))
            if len(xs) != 2 or len(ys) != 2:  # not an axis-aligned rect
                continue
            xc = 0.5 * (xs[0] + xs[1])
            by_color.setdefault(fill.lower(), []).append(
                (xc, ys[0], ys[1]))  # y_top (svg-min), y_base
    series = []
    for col, bars in by_color.items():
        bars.sort()
        series.append({"id": "bars_" + col.lstrip("#"), "color": col,
                       "kind": "bars",
                       "pts": [(x, ytop) for x, ytop, ybase in bars]})
    return series


def extract_legend(root):
    """All legends in figure: ordered (color, label) pairs."""
    mapping = []  # list of (color, label)
    for leg, chain in find_groups(root, "legend_"):
        pending_color = None
        for el, _ in iter_with_parents(leg):
            if callable(el.tag) or tag_name(el) != "g":
                continue
            eid = el.get("id", "")
            if eid.startswith(("line2d_", "PathCollection_", "patch_")):
                col = None
                for sub, _ in iter_with_parents(el):
                    if callable(sub.tag):
                        continue
                    if tag_name(sub) in ("path", "use"):
                        c = style_get(sub, "stroke") or style_get(sub, "fill")
                        if c and c != "none":
                            col = c
                            break
                if col:
                    pending_color = col.lower()
            elif eid.startswith("text_"):
                lab = text_of_group(el)
                if lab and pending_color:
                    mapping.append((pending_color, lab))
                    pending_color = None
                elif lab:
                    mapping.append((None, lab))
    return mapping


def free_texts(axes_g):
    """Text groups directly under the axes (labels, titles, annotations),
    excluding tick labels and legend text. Returns [(text, (tx, ty), rotated)]"""
    out = []
    for el, chain in iter_with_parents(axes_g):
        if callable(el.tag) or tag_name(el) != "g":
            continue
        if not el.get("id", "").startswith("text_"):
            continue
        if any(c and (c.startswith("xtick") or c.startswith("ytick") or
                      c.startswith("legend")) for c in chain):
            continue
        txt = text_of_group(el)
        if not txt:
            continue
        tx = ty = None
        rot = False
        for sub in el:
            if callable(sub.tag):
                continue
            tr = sub.get("transform", "")
            m = re.search(r"translate\(([-\d.]+)\s+([-\d.]+)\)", tr)
            if m:
                tx, ty = float(m.group(1)), float(m.group(2))
            if "rotate" in tr:
                rot = True
        out.append((txt, (tx, ty), rot))
    return out


def classify_texts(texts, bbox):
    """Guess xlabel/ylabel/title from positions relative to axes bbox."""
    xlabel = ylabel = title = None
    other = []
    if bbox is None:
        return None, None, None, [t for t, _, _ in texts]
    x0, y0, x1, y1 = bbox
    for txt, (tx, ty), rot in texts:
        if tx is None:
            other.append(txt)
        elif rot and (tx < x0 or tx > x1) and ylabel is None:
            ylabel = txt  # rotated text outside left/right edge
        elif ty is not None and ty > y1 and x0 <= tx <= x1 + 20:
            xlabel = txt
        elif ty is not None and ty < y0:
            title = txt
        else:
            other.append(txt)
    return xlabel, ylabel, title, other


# ----------------------------------------------------------------------
# main driver
# ----------------------------------------------------------------------

def sanitize(s, maxlen=40):
    s = re.sub(r"[^\w.+-]+", "_", s).strip("_")
    return s[:maxlen] if s else "unknown"


def find_offset_multiplier(axes_g, kind):
    """Look for an axis offset text (e.g. '1e-9') inside matplotlib.axis_*
    groups that is NOT inside a tick group."""
    for el, chain in iter_with_parents(axes_g):
        if callable(el.tag) or tag_name(el) != "g":
            continue
        if not el.get("id", "").startswith("text_"):
            continue
        in_axis = any(c and c.startswith("matplotlib.axis") for c in chain)
        in_tick = any(c and (c.startswith("xtick") or c.startswith("ytick"))
                      for c in chain)
        if in_axis and not in_tick:
            mult = parse_offset_text(text_of_group(el))
            if mult is not None:
                return mult
    return None


def digitize_figure(figname):
    svg_path = os.path.join(IN_DIR, figname + ".svg")
    root = parse_svg(svg_path)
    legend = extract_legend(root)
    fig_report = {"svg": svg_path, "legend": [
        {"color": c, "label": l} for c, l in legend], "axes": []}

    axes_groups = [(g, ch) for g, ch in find_groups(root, "axes_")
                   if re.match(r"axes_\d+$", g.get("id", ""))]
    axes_groups.sort(key=lambda t: int(t[0].get("id").split("_")[1]))

    # first pass: build tick maps for all axes (for shared-axis fallback)
    axinfo = []
    for g, _ in axes_groups:
        bbox = axes_bbox(g)
        xticks = collect_ticks(g, "xtick")
        yticks = collect_ticks(g, "ytick")
        axinfo.append({
            "g": g, "id": g.get("id"), "bbox": bbox,
            "xticks": xticks, "yticks": yticks,
            "xmap": AxisMap(xticks, "x"), "ymap": AxisMap(yticks, "y"),
        })

    # shared-axis fallback: borrow a working map from an axes whose data
    # rect spans the same pixel range on that dimension
    for a in axinfo:
        for dim, lo, hi in (("xmap", 0, 2), ("ymap", 1, 3)):
            if a[dim].ok or a["bbox"] is None:
                continue
            for b in axinfo:
                if b is a or not b[dim].ok or b["bbox"] is None:
                    continue
                if (abs(a["bbox"][lo] - b["bbox"][lo]) < 1.0 and
                        abs(a["bbox"][hi] - b["bbox"][hi]) < 1.0):
                    a[dim] = b[dim]
                    a.setdefault("notes", []).append(
                        f"{dim} borrowed from {b['id']} (shared axis)")
                    break
    # twinx fallback: axes with yticks but no bbox and no xticks (twin
    # y-axis shares the donor's x axis and plot rectangle)
    for a in axinfo:
        if a["bbox"] is not None or a["xmap"].ok or not a["yticks"]:
            continue
        ay = [p for p, _, _ in a["yticks"]]
        best, best_ov = None, 0.0
        for b in axinfo:
            if b is a or not b["xmap"].ok or b["bbox"] is None:
                continue
            _, y0, _, y1 = b["bbox"]
            ov = sum(1 for p in ay if y0 - 2 <= p <= y1 + 2) / len(ay)
            if ov > best_ov:
                best, best_ov = b, ov
        if best is not None and best_ov > 0.8:
            a["xmap"] = best["xmap"]
            a["bbox"] = best["bbox"]
            a.setdefault("notes", []).append(
                f"xmap+bbox borrowed from {best['id']} (twin axes)")

    used_names = set()
    for a in axinfo:
        g = a["g"]
        aid = a["id"]
        bbox = a["bbox"]
        xmap, ymap = a["xmap"], a["ymap"]
        xmult = find_offset_multiplier(g, "x") or 1.0
        texts = free_texts(g)
        xlabel, ylabel, title, other = classify_texts(texts, bbox)
        curves = extract_curves(g)
        notes = a.get("notes", [])
        if any(not callable(el.tag) and tag_name(el) == "image"
               for el, _ in iter_with_parents(g)):
            notes.append("axes contains an embedded raster <image> "
                         "(photo/inset); not digitizable as vector data")
        if not xmap.ok:
            notes.append("no decodable x ticks: exporting pixel x")
        if not ymap.ok:
            notes.append("no decodable y ticks: exporting pixel y")

        entry = {
            "axes": aid,
            "bbox_px": bbox,
            "xlabel": xlabel, "ylabel": ylabel, "title": title,
            "other_texts": other,
            "xscale": xmap.scale or "pixel",
            "yscale": ymap.scale or "pixel",
            "xticks": [{"px": round(p, 2), "label": lab, "value": v}
                       for p, v, lab in a["xticks"] if lab],
            "yticks": [{"px": round(p, 2), "label": lab, "value": v}
                       for p, v, lab in a["yticks"] if lab],
            "notes": notes,
            "curves": [],
        }

        color2label = {}
        for c, l in legend:
            if c and c not in color2label:
                color2label[c] = l

        for ci, cur in enumerate(curves):
            px = np.array([p[0] for p in cur["pts"]])
            py = np.array([p[1] for p in cur["pts"]])
            xd = xmap.to_data(px) if xmap.ok else px
            yd = ymap.to_data(py) if ymap.ok else py
            label = color2label.get(cur["color"])
            tag = sanitize(label) if label else cur["color"].lstrip("#")
            base = f"{figname.rstrip('_')}_{aid}_c{ci}_{tag}"
            name = base
            k = 1
            while name in used_names:
                name = f"{base}_{k}"
                k += 1
            used_names.add(name)
            csv_path = os.path.join(OUT_DIR, name + ".csv")
            with open(csv_path, "w") as f:
                f.write("x,y\n")
                for xx, yy in zip(xd, yd):
                    f.write(f"{xx:.6g},{yy:.6g}\n")
            crec = {
                "svg_id": cur["id"], "color": cur["color"],
                "kind": cur["kind"], "label": label,
                "n_points": int(len(px)),
                "x_range": [float(np.min(xd)), float(np.max(xd))],
                "y_range": [float(np.min(yd)), float(np.max(yd))],
                "csv": os.path.relpath(csv_path, OUT_DIR),
            }
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                frac_out = float(np.mean(
                    (px < x0 - 2) | (px > x1 + 2) |
                    (py < y0 - 2) | (py > y1 + 2)))
                if frac_out > 0.5:
                    crec["outside_axes"] = True
                    crec["note"] = ("most points outside plot rect; "
                                    "likely an annotation, not data")
            entry["curves"].append(crec)
        fig_report["axes"].append(entry)
    return fig_report


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    for fig in FIGS:
        try:
            manifest[fig] = digitize_figure(fig)
        except Exception as e:  # keep going, record failure
            import traceback
            manifest[fig] = {"error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()}
            print(f"FAILED {fig}: {e}", file=sys.stderr)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    # ---- sanity check: F2_ axes_1 ----
    f2 = manifest.get("F2_", {})
    for ax in f2.get("axes", []):
        if ax["axes"] == "axes_1":
            cs = ax["curves"]
            if cs:
                xr = [min(c["x_range"][0] for c in cs),
                      max(c["x_range"][1] for c in cs)]
                yr = [min(c["y_range"][0] for c in cs),
                      max(c["y_range"][1] for c in cs)]
                print(f"SANITY F2_ axes_1: x range {xr}  "
                      f"y range {yr}  n_curves={len(cs)}  "
                      f"xscale={ax['xscale']} yscale={ax['yscale']}")
                print(f"  xlabel={ax['xlabel']!r} ylabel={ax['ylabel']!r}")
            else:
                print("SANITY F2_ axes_1: NO CURVES FOUND")

    # brief summary
    for fig, rep in manifest.items():
        if "error" in rep:
            print(f"{fig}: ERROR {rep['error']}")
            continue
        for ax in rep["axes"]:
            print(f"{fig} {ax['axes']}: {len(ax['curves'])} curves, "
                  f"xscale={ax['xscale']} yscale={ax['yscale']}, "
                  f"xlabel={ax['xlabel']!r}, ylabel={ax['ylabel']!r}, "
                  f"notes={ax['notes']}")


if __name__ == "__main__":
    main()
