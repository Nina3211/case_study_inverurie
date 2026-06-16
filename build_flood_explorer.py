"""
build_flood_explorer.py
=======================
Flood explorer HTML / data-file builder for the UQ4FM probabilistic flood
mapping dashboard.

Usage (from the notebook):
    from build_flood_explorer import export_leaflet_html
    export_leaflet_html(
        prob_inundation = prob_inundation_joint,
        depth_p05       = depth_p05_joint,
        depth_p10       = depth_p10_joint,
        depth_p25       = depth_p25_joint,
        depth_p50       = depth_p50_joint,
        depth_p75       = depth_p75_joint,
        depth_p90       = depth_p90_joint,
        depth_p95       = depth_p95_joint,
        all_mean_maps   = all_mean_maps_approx,
        det_map         = det_map,
        dem_arr         = dem_arr,
        det_map_CC      = det_map_CC,
        xll=xll, yll=yll, cs=cs,
        site_name       = SITE_NAME,
        return_period   = RETURN_PERIOD,
        n_mc            = N_MC,
        **{k: extent[k] for k in ['lon_min','lon_max','lat_min','lat_max']},
        out_path        = os.path.join(FIGURES_DIR, "flood_explorer.html"),
    )

Data storage — three modes
--------------------------
external_data='embed'  (DEFAULT)
    Arrays are gzip-compressed and base64-encoded directly inside the HTML
    in hidden <script> tags.  The JS runtime decompresses them with the
    browser's native DecompressionStream API.  Works locally (file://) AND
    on GitHub Pages with no extra files.  HTML will be ~30-60 MB depending
    on the grid — well within GitHub Pages limits.

external_data='files'
    Arrays are written to <out_path>.data/*.json.gz and fetched at runtime.
    Keeps HTML tiny (~50 KB) but requires an HTTP server — will NOT work
    when opening the HTML directly from disk (file:// CORS error).
    Use this mode only for GitHub Pages deployment (commit the .data/ folder).

external_data=False
    All arrays embedded as raw (uncompressed) JSON strings inline.
    Largest output, but works everywhere and needs no extra files.
    Useful as a fallback if the browser lacks DecompressionStream support.
"""

import json
import os
import gzip
import base64
import io
import numpy as np

# ---------------------------------------------------------------------------
# Leaflet assets (CDN — no local copies needed)
# ---------------------------------------------------------------------------
_LEAFLET_CSS = '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>'
_LEAFLET_JS  = '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _j(arr: np.ndarray) -> str:
    """Round-trip array to compact JSON string (2 dp)."""
    return json.dumps(np.round(arr.astype(float), 2).tolist())


def _j_gz_b64(arr: np.ndarray) -> str:
    """Gzip-compress a JSON-serialised array and return as a base64 string.
    This is ~3-8× smaller than raw JSON and works in any browser without fetch().
    """
    raw = _j(arr).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
        gz.write(raw)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _save_gzip(path: str, data: str) -> None:
    """Write a string as gzip-compressed UTF-8 (for external_data='files' mode)."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(data)

def _compute_buckets(arrays_depth, array_prob, array_range, depth_thresh=0.05, n_buckets=6):
    """
    Compute discrete bucket breakpoints.

    Probability: fixed meaningful thresholds for flood risk communication.
    Depth / range: data-driven from wet-cell percentiles so buckets are
                   always well-distributed regardless of site or return period.

    Returns a dict with keys:
      depth  : list of n_buckets+1 edges (shared across all depth layers)
      prob   : list of n_buckets+1 edges (fixed)
      range  : list of n_buckets+1 edges (data-driven)
    """
    # ── Probability: fixed thresholds ──────────────────────────────────────
    prob_edges = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]  # 6 buckets

    # ── Depth / range: data-driven from wet-cell percentiles ───────────────
    pcts = [i * (100 / (n_buckets - 1)) for i in range(n_buckets - 1)]

    def edges_from(arr, thresh=0.0):
        flat = arr.ravel()
        wet  = flat[flat > thresh]
        if len(wet) < 10:
            return [0.0] * (n_buckets + 1)
        pts = np.percentile(wet, pcts + [99.0])
        # Round to 2 dp
        pts = [round(float(p), 2) for p in pts]
        # Ensure strictly increasing from 0
        result = [0.0]
        for v in pts:
            if v > result[-1]:
                result.append(v)
        while len(result) < n_buckets + 1:
            result.append(round(result[-1] + 0.01, 2))
        return result[:n_buckets + 1]

    # Depth: compute from element-wise max across all provided depth arrays
    depth_max = np.zeros_like(arrays_depth[0], dtype=float)
    for arr in arrays_depth:
        depth_max = np.maximum(depth_max, arr.astype(float))

    return {
        'depth': edges_from(depth_max, thresh=depth_thresh),
        'prob':  prob_edges,
        'range': edges_from(array_range, thresh=depth_thresh),
    }




# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------
def export_leaflet_html(
    # ── Required probabilistic layers ──────────────────────────────────────
    prob_inundation,
    depth_p05,
    depth_p50,
    depth_p95,
    all_mean_maps,           # shape (n_quantile_levels, NROWS, NCOLS)
    # ── Additional quantile layers (for exceedance curve) ──────────────────
    depth_p10=None,
    depth_p25=None,
    depth_p75=None,
    depth_p90=None,
    # ── Optional deterministic / DEM layers ────────────────────────────────
    det_map=None,
    dem_arr=None,
    det_map_CC=None,
    # ── Grid geometry ───────────────────────────────────────────────────────
    xll=None, yll=None, cs=None,
    # ── Metadata ────────────────────────────────────────────────────────────
    site_name="Inverurie",
    return_period=100,
    n_mc=1000,
    # ── Bounding box ────────────────────────────────────────────────────────
    lon_min=-2.415, lon_max=-2.345,
    lat_min=57.268, lat_max=57.302,
    # ── Appearance ──────────────────────────────────────────────────────────
    overlay_alpha=0.60,
    # ── Output ──────────────────────────────────────────────────────────────
    out_path="flood_explorer.html",
    external_data='embed',    # 'embed' | 'files' | False
    about_page_url="about.html",
):
    """
    Generate the interactive flood explorer HTML dashboard.

    Parameters
    ----------
    prob_inundation : ndarray (NROWS, NCOLS)
    depth_p05 … depth_p95 : ndarray (NROWS, NCOLS)
    all_mean_maps : ndarray (n_levels, NROWS, NCOLS)
        Stack of quantile grids used to derive exceedance curves.
        Ideally pass all 7: p05, p10, p25, p50, p75, p90, p95.
    det_map : ndarray or None
        Deterministic flood depth map.
    dem_arr : ndarray or None
        DEM for cross-section tab.
    det_map_CC : ndarray or None
        Climate-change deterministic depth map.
    external_data : str or bool
        'embed'  (default) — gzip+base64 data embedded in HTML; works locally
                             AND on GitHub Pages, no extra files needed.
        'files'  — write .json.gz files to <out_path>.data/; requires HTTP
                   server (will fail when opened from disk via file://).
        False    — embed raw JSON inline; largest output, works everywhere.
    about_page_url : str
        URL of the companion "About the emulator" page.
    """

    NROWS, NCOLS = prob_inundation.shape
    clat = (lat_min + lat_max) / 2
    clon = (lon_min + lon_max) / 2

    vmax_depth = float(max(
        np.nanmax(depth_p05),
        np.nanmax(depth_p50),
        np.nanmax(depth_p95),
        np.nanmax(det_map) if det_map is not None else 0,
    ))
    print(f"Shared colour scale: 0 → {vmax_depth:.3f} m  (max across all depth layers)")

    dLat = (lat_max - lat_min) / (NROWS - 1)
    dLon = (lon_max - lon_min) / (NCOLS - 1)
    img_lat_min = lat_min - dLat / 2
    img_lat_max = lat_max + dLat / 2
    img_lon_min = lon_min - dLon / 2
    img_lon_max = lon_max + dLon / 2

    # ── Quantile levels for exceedance curve ───────────────────────────────
    # Build a stack that includes all available percentile layers.
    # PROB_LEVELS order must match the stack axis-0 order.
    _quant_stack_parts = []
    _prob_levels       = []
    for prob_val, arr_val in [
        (0.05, depth_p05),
        (0.10, depth_p10),
        (0.25, depth_p25),
        (0.50, depth_p50),
        (0.75, depth_p75),
        (0.90, depth_p90),
        (0.95, depth_p95),
    ]:
        if arr_val is not None:
            _quant_stack_parts.append(arr_val)
            _prob_levels.append(prob_val)

    if len(_quant_stack_parts) >= 3:
        quant_depths = np.stack(_quant_stack_parts, axis=0)
    else:
        # Fall back to percentile computation from all_mean_maps
        _prob_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
        quant_depths = np.percentile(
            all_mean_maps, [p * 100 for p in _prob_levels], axis=0)

    # ── Range map: p95 − median ─────────────────────────────────────────────
    range_p95_median = np.maximum(depth_p95 - depth_p50, 0)
    range_p95_p05    = np.maximum(depth_p95 - depth_p05, 0)

    # ── Discrete buckets (data-driven, from wet-cell percentiles) ──────────────
    _depth_arrays = [depth_p05, depth_p50, depth_p95]
    if det_map    is not None: _depth_arrays.append(det_map)
    if det_map_CC is not None: _depth_arrays.append(det_map_CC)
    _buckets = _compute_buckets(
        arrays_depth = _depth_arrays,
        array_prob   = prob_inundation,
        array_range  = np.maximum(range_p95_p05, range_p95_median),
    )
    print(f"Depth buckets:  {_buckets['depth']}")
    print(f"Prob  buckets:  {_buckets['prob']}")
    print(f"Range buckets:  {_buckets['range']}")

    # ── True maxes for colorbar info ────────────────────────────────────────
    det_true_max = f"{float(np.nanmax(det_map)):.2f}" if det_map is not None else "null"
    p05_true_max = f"{float(np.nanmax(depth_p05)):.2f}"
    p50_true_max = f"{float(np.nanmax(depth_p50)):.2f}"
    p95_true_max = f"{float(np.nanmax(depth_p95)):.2f}"
    rng_med_max  = f"{float(np.nanmax(range_p95_median)):.2f}"
    rng_full_max = f"{float(np.nanmax(range_p95_p05)):.2f}"

    # ── Cross-section ────────────────────────────────────────────────────────
    xs_btn_html = ""
    if dem_arr is not None and xll is not None:
        det_cc_str = _j(np.maximum(det_map_CC, 0)) if det_map_CC is not None else "null"
        det_str    = _j(np.maximum(det_map,    0)) if det_map    is not None else "null"
        xs_data_inline = f"""
const XS_ENABLED = true;
const XLL={xll}, YLL={yll}, CS_M={cs};
const XS_DATA = {{
  dem:    {_j(dem_arr)},
  p05:    {_j(depth_p05)},
  p50:    {_j(depth_p50)},
  p95:    {_j(depth_p95)},
  det:    {det_str},
  det_cc: {det_cc_str},
}};"""
        xs_btn_html = """<button id="xs-btn" style="margin-left:16px;padding:4px 14px;
            background:#0f2236;color:#60a5fa;border:1px solid #1e3a5f;border-radius:5px;
            font-size:12px;cursor:pointer;" title="Toggle cross-section mode">
            ✂ Cross-section</button>"""
    else:
        xs_data_inline = "const XS_ENABLED = false;"

    # ── Deterministic layer options ──────────────────────────────────────────
    if det_map is not None:
        det_layer_entry = (
            f'"Deterministic (m)":{{data:{_j(np.maximum(det_map, 0))},'
            f'vmin:0,vmax:VMAX_DEPTH,cmap:"Blues",unit:" m"}},'
        )
        det_layer_cc_entry = (
            f'"Det. + climate change (m)":{{data:{_j(np.maximum(det_map_CC, 0) if det_map_CC is not None else np.zeros_like(det_map))},'
            f'vmin:0,vmax:VMAX_DEPTH,cmap:"Blues",unit:" m"}},'
        )
        det_option = '<option>Deterministic (m)</option><option>Det. + climate change (m)</option>'
        det_cc_true_max = (f"{float(np.nanmax(det_map_CC)):.2f}"
                           if det_map_CC is not None else "null")
        det_card = f"""
<div class="icard">
  <div class="icard-head">Deterministic run</div>
  <p>Single best-estimate flood depth from the calibrated hydraulic model using
  median inflow parameters. No uncertainty is represented — compare alongside the
  probabilistic layers to see how the central estimate sits within the ensemble spread.</p>
</div>"""
        det_toggle_btn = """
    <hr class="sep">
    <div class="panel-head" style="margin-bottom:6px;">Quick switch</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
      <button class="btn btn-active" id="prob-btn">Probabilistic</button>
      <button class="btn" id="det-btn">Deterministic</button>
    </div>"""
        det_layer_maxes = f"""
  "Deterministic (m)":         {det_true_max},
  "Det. + climate change (m)": {det_cc_true_max},"""
    else:
        det_layer_entry    = ""
        det_layer_cc_entry = ""
        det_option         = ""
        det_card           = ""
        det_toggle_btn     = ""
        det_layer_maxes    = ""

    n_info_cols = 7 if det_map is not None else 6

    # ── Collect all arrays we need to store ─────────────────────────────────
    _store = {
        "prob_inundation":  prob_inundation,
        "depth_p50":        depth_p50,
        "depth_p05":        depth_p05,
        "depth_p95":        depth_p95,
        "range_p95_median": range_p95_median,
        "range_p95_p05":    range_p95_p05,
        "quant_depths":     quant_depths,
    }
    if det_map is not None:
        _store["det_map"]    = np.maximum(det_map, 0)
        _store["det_map_cc"] = (np.maximum(det_map_CC, 0)
                                if det_map_CC is not None
                                else np.zeros_like(det_map))
    if dem_arr is not None:
        _store["dem_arr"] = dem_arr
        _store["xs_p05"]  = depth_p05
        _store["xs_p50"]  = depth_p50
        _store["xs_p95"]  = depth_p95

    # ── Mode: 'embed' — gzip+base64 inside the HTML ─────────────────────────
    if external_data == 'embed':
        print("Encoding arrays as gzip+base64 (embed mode)…")
        _b64 = {}
        total_kb = 0
        for key, arr in _store.items():
            enc = _j_gz_b64(arr)
            _b64[key] = enc
            total_kb += len(enc) * 3 / 4 / 1024   # approximate decoded bytes
            print(f"  {key}: {len(enc)*3//4//1024} KB compressed")
        print(f"Total embedded data: ~{total_kb:.0f} KB compressed")

        # Build the embedded-data script block
        embed_script_lines = []
        for key, b64str in _b64.items():
            embed_script_lines.append(
                f'<script id="data-{key}" type="application/gzip-b64">{b64str}</script>'
            )
        embedded_data_tags = "\n".join(embed_script_lines)

        data_loader_js = _build_embed_loader_js(
            det_map=det_map, dem_arr=dem_arr, det_map_CC=det_map_CC,
            xll=xll, yll=yll, cs=cs,
            range_p95_median_max=float(np.nanmax(range_p95_median)),
            range_p95_p05_max=float(np.nanmax(range_p95_p05)),
            vmax_depth=vmax_depth,
        )
        xs_data_js_block = ""
        layers_inline    = ""
        quant_inline     = ""

    # ── Mode: 'files' — external .json.gz files (HTTP only) ─────────────────
    elif external_data == 'files':
        data_dir = out_path + ".data"
        os.makedirs(data_dir, exist_ok=True)
        embedded_data_tags = (
            "<!-- external_data='files': data loaded via fetch() from .data/ dir -->\n"
            "<!-- NOTE: requires HTTP server; will NOT work with file:// URLs -->"
        )
        saved_kb = 0
        for key, arr in _store.items():
            fpath = os.path.join(data_dir, f"{key}.json.gz")
            _save_gzip(fpath, _j(arr))
            saved_kb += os.path.getsize(fpath) / 1024
            print(f"  Written {fpath}  ({os.path.getsize(fpath)/1024:.0f} KB)")
        print(f"Total external data: {saved_kb:.0f} KB")
        print("⚠  'files' mode: open via HTTP server, not file:// — see docs.")

        data_loader_js = _build_external_loader_js(
            det_map=det_map, dem_arr=dem_arr, det_map_CC=det_map_CC,
            data_dir_name=os.path.basename(data_dir),
            xll=xll, yll=yll, cs=cs,
        )
        xs_data_js_block = ""
        layers_inline    = ""
        quant_inline     = ""

    # ── Mode: False — raw inline JSON (works everywhere, large file) ─────────
    else:
        embedded_data_tags = ""
        data_loader_js = ""
        xs_data_js_block = xs_data_inline
        layers_inline = f"""
const LAYERS = {{
  {det_layer_entry}
  {det_layer_cc_entry}
  "P(inundation)":       {{data:{_j(prob_inundation)}, vmin:0, vmax:1,          cmap:"YlOrRd", unit:""}},
  "Median depth (m)":    {{data:{_j(depth_p50)},       vmin:0, vmax:VMAX_DEPTH, cmap:"Blues",  unit:" m"}},
  "P05 depth (m)":       {{data:{_j(depth_p05)},       vmin:0, vmax:VMAX_DEPTH, cmap:"Blues",  unit:" m"}},
  "P95 depth (m)":       {{data:{_j(depth_p95)},       vmin:0, vmax:VMAX_DEPTH, cmap:"Blues",  unit:" m"}},
  "P95\u2212Median (m)": {{data:{_j(range_p95_median)},vmin:0, vmax:{float(np.nanmax(range_p95_median)):.3f}, cmap:"RdPu",  unit:" m"}},
  "P95\u2212P05 (m)":    {{data:{_j(range_p95_p05)},   vmin:0, vmax:{float(np.nanmax(range_p95_p05)):.3f},   cmap:"RdPu",  unit:" m"}},
}};"""
        quant_inline = f"const QUANT_DEPTHS={_j(quant_depths)};"

    # ── static JS data block ─────────────────────────────────────────────────
    _xs_block = "" if external_data else xs_data_js_block

    # In embed/files mode we need placeholder var declarations so logic_js can
    # reference these globals before the async loader populates them.
    if external_data:
        async_placeholders = """
var LAYERS = {};
var QUANT_DEPTHS = null;
var XS_ENABLED = false;
var XS_DATA = null;
var XLL = null, YLL = null, CS_M = null;
"""
    else:
        async_placeholders = ""

    data_js = f"""
const NROWS={NROWS}, NCOLS={NCOLS};
const LAT_MIN={lat_min}, LAT_MAX={lat_max};
const LON_MIN={lon_min}, LON_MAX={lon_max};
const IMG_LAT_MIN={img_lat_min:.7f}, IMG_LAT_MAX={img_lat_max:.7f};
const IMG_LON_MIN={img_lon_min:.7f}, IMG_LON_MAX={img_lon_max:.7f};
const CLAT={clat}, CLON={clon};
const VMAX_DEPTH={vmax_depth:.4f};
const ALPHA_DEFAULT={overlay_alpha};
const PROB_LEVELS={json.dumps(_prob_levels)};
const BUCKETS={{
  depth:{json.dumps(_buckets['depth'])},
  prob: {json.dumps(_buckets['prob'])},
  range:{json.dumps(_buckets['range'])},
}};
{quant_inline}
{layers_inline}
const LAYER_MAXES = {{{det_layer_maxes}
  "P(inundation)":       null,
  "Median depth (m)":    {p50_true_max},
  "P05 depth (m)":       {p05_true_max},
  "P95 depth (m)":       {p95_true_max},
  "P95\u2212Median (m)": {rng_med_max},
  "P95\u2212P05 (m)":    {rng_full_max}
}};
{async_placeholders}
{_xs_block}
"""

    logic_js = _build_logic_js(return_period=return_period, n_mc=n_mc,
                                det_map=det_map)

    info_cards = _build_info_cards(return_period, n_mc, det_card)

    html = _build_html(
        site_name=site_name,
        return_period=return_period,
        n_mc=n_mc,
        overlay_alpha=overlay_alpha,
        xs_btn_html=xs_btn_html,
        det_option=det_option,
        det_toggle_btn=det_toggle_btn,
        n_info_cols=n_info_cols,
        info_cards=info_cards,
        data_js=data_js,
        logic_js=logic_js,
        data_loader_js=data_loader_js,
        embedded_data_tags=embedded_data_tags,
        about_page_url=about_page_url,
        leaflet_css=_LEAFLET_CSS,
        leaflet_js=_LEAFLET_JS,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved → {out_path}  ({size_kb:.0f} KB)")
    if external_data == 'files':
        print(f"Data files → {out_path}.data/")
    return out_path


# ---------------------------------------------------------------------------
# Embed loader JS (gzip+base64 in <script> tags — works file:// and https://)
# ---------------------------------------------------------------------------
def _build_embed_loader_js(det_map, dem_arr, det_map_CC,
                            xll, yll, cs,
                            range_p95_median_max, range_p95_p05_max,
                            vmax_depth):
    has_det = det_map is not None
    has_xs  = dem_arr is not None and xll is not None

    det_keys = '"det_map","det_map_cc",' if has_det else ''
    xs_keys  = '"dem_arr","xs_p05","xs_p50","xs_p95",' if has_xs else ''

    xs_init = f"""
    XS_ENABLED = true;
    XLL = {xll}; YLL = {yll}; CS_M = {cs};
    XS_DATA = {{
      dem:    d.dem_arr,
      p05:    d.xs_p05,
      p50:    d.xs_p50,
      p95:    d.xs_p95,
      det:    {("d.det_map" if has_det else "null")},
      det_cc: {("d.det_map_cc" if has_det else "null")},
    }};""" if has_xs else "XS_ENABLED = false;"

    det_layers = """
    LAYERS["Deterministic (m)"]         = {data: d.det_map,    vmin:0, vmax:VMAX_DEPTH, cmap:"Blues", unit:" m"};
    LAYERS["Det. + climate change (m)"] = {data: d.det_map_cc, vmin:0, vmax:VMAX_DEPTH, cmap:"Blues", unit:" m"};""" if has_det else ""

    return f"""
// ── Embedded gzip+base64 data loader ─────────────────────────────────────
(function() {{
  var KEYS = [
    "prob_inundation","depth_p50","depth_p05","depth_p95",
    "range_p95_median","range_p95_p05","quant_depths",
    {det_keys}
    {xs_keys}
  ];

  // Decompress a base64-gzip string to a parsed JS value.
  // Uses native DecompressionStream (Chrome 80+, Firefox 113+, Edge 80+, Safari 16.4+).
  // Falls back to a pure-JS inflate for older browsers.
  function decodeB64Gz(b64, callback) {{
    var binary = atob(b64);
    var bytes  = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    if (typeof DecompressionStream !== 'undefined') {{
      var ds  = new DecompressionStream('gzip');
      var w   = ds.writable.getWriter();
      var r   = ds.readable.getReader();
      var chunks = [];
      function pump() {{
        r.read().then(function(res) {{
          if (res.done) {{
            var blob    = new Blob(chunks);
            blob.text().then(function(txt) {{ callback(JSON.parse(txt)); }});
          }} else {{
            chunks.push(res.value);
            pump();
          }}
        }});
      }}
      pump();
      w.write(bytes);
      w.close();
    }} else {{
      // Minimal fallback: raw inflate via Response (works in most modern browsers)
      new Response(new Blob([bytes])).arrayBuffer().then(function(buf) {{
        // last resort — try pako if available, otherwise alert
        if (typeof pako !== 'undefined') {{
          callback(JSON.parse(pako.inflate(new Uint8Array(buf), {{to:'string'}})));
        }} else {{
          alert('Your browser does not support DecompressionStream. ' +
                'Please use Chrome, Firefox, Edge, or Safari 16.4+.');
        }}
      }});
    }}
  }}

  var d = {{}}, remaining = KEYS.length;
  function onDone() {{
    if (--remaining > 0) return;
    LAYERS["P(inundation)"]       = {{data: d.prob_inundation, vmin:0, vmax:1,          cmap:"YlOrRd", unit:""}};
    LAYERS["Median depth (m)"]    = {{data: d.depth_p50,       vmin:0, vmax:VMAX_DEPTH, cmap:"Blues",  unit:" m"}};
    LAYERS["P05 depth (m)"]       = {{data: d.depth_p05,       vmin:0, vmax:VMAX_DEPTH, cmap:"Blues",  unit:" m"}};
    LAYERS["P95 depth (m)"]       = {{data: d.depth_p95,       vmin:0, vmax:VMAX_DEPTH, cmap:"Blues",  unit:" m"}};
    LAYERS["P95\u2212Median (m)"] = {{data: d.range_p95_median, vmin:0, vmax:{range_p95_median_max:.3f}, cmap:"RdPu", unit:" m"}};
    LAYERS["P95\u2212P05 (m)"]    = {{data: d.range_p95_p05,   vmin:0, vmax:{range_p95_p05_max:.3f},   cmap:"RdPu", unit:" m"}};{det_layers}
    QUANT_DEPTHS = d.quant_depths;
    {xs_init}
    document.getElementById('loading-overlay').style.display = 'none';
    initMap();
  }}

  KEYS.forEach(function(key) {{
    var el = document.getElementById('data-' + key);
    if (!el) {{ remaining--; if (remaining === 0) onDone(); return; }}
    decodeB64Gz(el.textContent.trim(), function(val) {{
      d[key] = val;
      onDone();
    }});
  }});
}})();
"""


# ---------------------------------------------------------------------------
# External data loader JS (fetch from .data/ — HTTP only)
# ---------------------------------------------------------------------------
def _build_external_loader_js(det_map, dem_arr, det_map_CC, data_dir_name,
                               xll, yll, cs):
    has_det = det_map is not None
    has_xs  = dem_arr is not None and xll is not None

    det_load = f"""
    ["det_map",    "{data_dir_name}/det_map.json.gz"],
    ["det_map_cc", "{data_dir_name}/det_map_cc.json.gz"],""" if has_det else ""

    xs_load = f"""
    ["dem_arr", "{data_dir_name}/dem_arr.json.gz"],
    ["xs_p05",  "{data_dir_name}/xs_p05.json.gz"],
    ["xs_p50",  "{data_dir_name}/xs_p50.json.gz"],
    ["xs_p95",  "{data_dir_name}/xs_p95.json.gz"],""" if has_xs else ""

    xs_init = f"""
    XS_ENABLED = true;
    XLL = {xll}; YLL = {yll}; CS_M = {cs};
    XS_DATA = {{
      dem:    d.dem_arr,
      p05:    d.xs_p05,
      p50:    d.xs_p50,
      p95:    d.xs_p95,
      det:    {("d.det_map" if has_det else "null")},
      det_cc: {("d.det_map_cc" if has_det else "null")},
    }};""" if has_xs else "XS_ENABLED = false;"

    det_layers = """
    LAYERS["Deterministic (m)"]          = {data: d.det_map,    vmin:0, vmax:VMAX_DEPTH, cmap:"Blues", unit:" m"};
    LAYERS["Det. + climate change (m)"]  = {data: d.det_map_cc, vmin:0, vmax:VMAX_DEPTH, cmap:"Blues", unit:" m"};""" if has_det else ""

    return f"""
// ── External data loader ──────────────────────────────────────────────────
(function() {{
  var FILES = [
    ["prob_inundation",  "{data_dir_name}/prob_inundation.json.gz"],
    ["depth_p50",        "{data_dir_name}/depth_p50.json.gz"],
    ["depth_p05",        "{data_dir_name}/depth_p05.json.gz"],
    ["depth_p95",        "{data_dir_name}/depth_p95.json.gz"],
    ["range_p95_median", "{data_dir_name}/range_p95_median.json.gz"],
    ["range_p95_p05",    "{data_dir_name}/range_p95_p05.json.gz"],
    ["quant_depths",     "{data_dir_name}/quant_depths.json.gz"],{det_load}{xs_load}
  ];
  var d = {{}}, pending = FILES.length;
  function done() {{
    if(--pending > 0) return;
    // Populate LAYERS
    window.LAYERS = window.LAYERS || {{}};
    LAYERS["P(inundation)"]       = {{data: d.prob_inundation, vmin:0, vmax:1,          cmap:"YlOrRd", unit:""}};
    LAYERS["Median depth (m)"]    = {{data: d.depth_p50,       vmin:0, vmax:VMAX_DEPTH, cmap:"Blues",  unit:" m"}};
    LAYERS["P05 depth (m)"]       = {{data: d.depth_p05,       vmin:0, vmax:VMAX_DEPTH, cmap:"Blues",  unit:" m"}};
    LAYERS["P95 depth (m)"]       = {{data: d.depth_p95,       vmin:0, vmax:VMAX_DEPTH, cmap:"Blues",  unit:" m"}};
    LAYERS["P95\u2212Median (m)"] = {{data: d.range_p95_median,vmin:0, vmax:LAYER_MAXES["P95\u2212Median (m)"], cmap:"RdPu",  unit:" m"}};
    LAYERS["P95\u2212P05 (m)"]    = {{data: d.range_p95_p05,   vmin:0, vmax:LAYER_MAXES["P95\u2212P05 (m)"],   cmap:"RdPu",  unit:" m"}};{det_layers}
    window.QUANT_DEPTHS = d.quant_depths;
    {xs_init}
    document.getElementById('loading-overlay').style.display = 'none';
    initMap();
  }}
  FILES.forEach(function(pair) {{
    fetch(pair[1])
      .then(function(r){{ return r.arrayBuffer(); }})
      .then(function(buf) {{
        return new Response(
          new DecompressionStream ? new Response(buf).body.pipeThrough(new DecompressionStream('gzip')) : buf
        ).text();
      }})
      .then(function(txt) {{ d[pair[0]] = JSON.parse(txt); done(); }})
      .catch(function(e) {{ console.error("Failed to load "+pair[1], e); }});
  }});
}})();
"""



# ---------------------------------------------------------------------------
# Logic JS (canvas rendering, interactions)
# ---------------------------------------------------------------------------
def _build_logic_js(return_period, n_mc, det_map):
    has_det = det_map is not None
    return r"""
const CMAPS = {
  YlOrRd: [[255,255,204],[255,237,160],[254,217,118],[254,178,76],
            [253,141,60],[252,78,42],[227,26,28],[189,0,38],[128,0,38]],
  Blues:  [[247,251,255],[222,235,247],[198,219,239],[158,202,225],
            [107,174,214],[66,146,198],[33,113,181],[8,81,156],[8,48,107]],
  RdPu:   [[255,247,243],[253,224,221],[252,197,192],[250,159,181],
            [247,104,161],[221,52,151],[174,1,126],[122,1,119],[73,0,106]]
};
function lerpColor(cmap,t){
  var s=CMAPS[cmap],i=t*(s.length-1),lo=Math.floor(i),
      hi=Math.min(lo+1,s.length-1),f=i-lo;
  return s[lo].map(function(v,k){return Math.round(v+f*(s[hi][k]-v));});
}
// Discrete colour for a value given a bucket edge list + cmap
function bucketColor(cmap,edges,v){
  for(var i=0;i<edges.length-1;i++){
    if(v<=edges[i+1]||i===edges.length-2){
      var t=(i+0.5)/(edges.length-2);
      return lerpColor(cmap,t);
    }
  }
  return lerpColor(cmap,1);
}

var currentLayer='P(inundation)', opacity=ALPHA_DEFAULT;
var depthThresh=0.05, inundThresh=0.0, showLatLon=false, lastClickedCell=null;
var xsMode=false, xsPts=[], xsMarkers=[], xsLine=null, xsYWindow=5;
var discreteMode=false;
var prevProbOpacity=ALPHA_DEFAULT;  // remember opacity when switching to deterministic

// Layer type helpers
function layerIsDet(name){
  return name==='Deterministic (m)'||name==='Det. + climate change (m)';
}
function layerIsDepth(name){
  return name==='Median depth (m)'||name==='P05 depth (m)'||name==='P95 depth (m)'||layerIsDet(name);
}
function layerIsRange(name){
  return name.indexOf('P95\u2212')===0;
}
function bucketEdgesFor(name){
  if(name==='P(inundation)') return BUCKETS.prob;
  if(layerIsRange(name))     return BUCKETS.range;
  return BUCKETS.depth;
}
function cmapFor(name){ return LAYERS[name]?LAYERS[name].cmap:'Blues'; }

// ── Map ────────────────────────────────────────────────────────────────────
function initMap(){
  window._map = L.map('map',{center:[CLAT,CLON],zoom:14});
  L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',{
    attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains:'abcd', maxZoom:20
  }).addTo(_map);

  L.rectangle([[LAT_MIN,LON_MIN],[LAT_MAX,LON_MAX]],{
    color:'#f97316',weight:3,fill:false,dashArray:'10,5',opacity:0.95
  }).addTo(_map).bindTooltip('Study area boundary',{direction:'top'});

  window.offCanvas=document.createElement('canvas');
  offCanvas.width=NCOLS; offCanvas.height=NROWS;
  window.offCtx=offCanvas.getContext('2d');

  window.imgBounds=[[IMG_LAT_MIN,IMG_LON_MIN],[IMG_LAT_MAX,IMG_LON_MAX]];
  window.overlay=L.imageOverlay(renderCanvas(),imgBounds,
    {opacity:1,interactive:false,zIndex:200}).addTo(_map);

  // ── Context-sensitive legend ────────────────────────────────────────────
  window.legendCtrl = L.control({position:'topleft'});
  legendCtrl.onAdd = function(){
    window.legendDiv = L.DomUtil.create('div','flood-legend');
    updateLegendContent();
    return legendDiv;
  };
  window.legendOnMap = false;
  syncLegend();

  drawColorbar(); syncToggleBtns();

  // ── Map clicks ─────────────────────────────────────────────────────────
  _map.on('click',function(e){
    var lat=e.latlng.lat, lon=e.latlng.lng;
    if(lat<LAT_MIN||lat>LAT_MAX||lon<LON_MIN||lon>LON_MAX) return;

    if(xsMode){
      var pt=latLonToRowCol(lat,lon);
      if(xsPts.length===0||xsPts.length===2){
        xsPts=[]; xsMarkers.forEach(function(m){_map.removeLayer(m);}); xsMarkers=[];
        if(xsLine){_map.removeLayer(xsLine); xsLine=null;}
        xsPts.push(pt);
        xsMarkers.push(L.circleMarker([lat,lon],
          {radius:7,color:'#22c55e',fillColor:'#22c55e',fillOpacity:1}).addTo(_map));
        document.getElementById('xs-status').textContent='START set — now click END point';
      } else if(xsPts.length===1){
        xsPts.push(pt);
        xsMarkers.push(L.circleMarker([lat,lon],
          {radius:7,color:'#ef4444',fillColor:'#ef4444',fillOpacity:1}).addTo(_map));
        var ll0=cellToLatLon(xsPts[0].r,xsPts[0].c);
        xsLine=L.polyline([[ll0.lat,ll0.lon],[lat,lon]],
          {color:'#ef4444',weight:2,dashArray:'6,4'}).addTo(_map);
        document.getElementById('xs-status').textContent='Drawing...';
        drawXsProfile(xsPts[0],xsPts[1]);
      }
      return;
    }

    var r=Math.min(Math.max(Math.round((LAT_MAX-lat)/(LAT_MAX-LAT_MIN)*(NROWS-1)),0),NROWS-1);
    var c=Math.min(Math.max(Math.round((lon-LON_MIN)/(LON_MAX-LON_MIN)*(NCOLS-1)),0),NCOLS-1);
    lastClickedCell={r:r,c:c};
    var vp =LAYERS['P(inundation)'].data[r][c];
    var v50=LAYERS['Median depth (m)'].data[r][c];
    var v05=LAYERS['P05 depth (m)'].data[r][c];
    var v95=LAYERS['P95 depth (m)'].data[r][c];
    var vdet  =(LAYERS['Deterministic (m)'])          ?LAYERS['Deterministic (m)'].data[r][c]          :null;
    var vdetcc=(LAYERS['Det. + climate change (m)'])  ?LAYERS['Det. + climate change (m)'].data[r][c]  :null;

    // Build popup rows — bold+highlight the row matching the active layer
    function prow(layerName, label, val, unit){
      var isActive=(layerName===currentLayer);
      var bold =isActive?'font-weight:700;':'';
      var bg   =isActive?'background:rgba(59,130,246,0.15);border-radius:4px;padding:1px 3px;':'';
      var vStr =val!==null?(typeof val==='number'?val.toFixed(3):val)+''+unit:'—';
      return '<div class="pr" style="'+bg+'">'+
             '<span class="pk" style="'+bold+'">'+label+'</span>'+
             '<span class="pv" style="'+bold+'color:#60a5fa;">'+vStr+'</span></div>';
    }

    // Fixed rows
    var rows = prow('P(inundation)',          'P(inundation)',   vp,   '')
             + prow('Median depth (m)',        'Median depth',    v50,  ' m')
             + prow('P05 depth (m)',           'P05 depth',       v05,  ' m')
             + prow('P95 depth (m)',           'P95 depth',       v95,  ' m');
    if(vdet   !==null) rows+=prow('Deterministic (m)',          'Det. depth',      vdet,   ' m');
    if(vdetcc !==null) rows+=prow('Det. + climate change (m)',  'Det.+CC depth',   vdetcc, ' m');

    // Extra row for active layer if not already covered
    var coveredLayers=['P(inundation)','Median depth (m)','P05 depth (m)','P95 depth (m)',
                       'Deterministic (m)','Det. + climate change (m)'];
    if(coveredLayers.indexOf(currentLayer)<0 && LAYERS[currentLayer]){
      var vExtra=LAYERS[currentLayer].data[r][c];
      rows+=prow(currentLayer, currentLayer, vExtra, LAYERS[currentLayer].unit||'');
    }

    L.popup({maxWidth:270}).setLatLng(e.latlng).setContent(
      '<div style="padding:4px 2px;min-width:210px;">'+
      '<div style="font-weight:600;margin-bottom:7px;color:#e2e8f0;font-size:13px;">'+cellLabel(r,c)+'</div>'+
      rows+
      '</div>'
    ).openOn(_map);
    drawExceedance(r,c);
  });

  // ── Controls ───────────────────────────────────────────────────────────
  document.getElementById('layer-select').addEventListener('change',function(e){
    var prev=currentLayer;
    currentLayer=e.target.value;
    // Auto-opacity: 100% for deterministic, restore saved for probabilistic
    if(layerIsDet(currentLayer)&&!layerIsDet(prev)){
      prevProbOpacity=opacity;
      opacity=1.0;
      document.getElementById('opacity-slider').value=100;
      document.getElementById('opacity-val').textContent='100';
    } else if(!layerIsDet(currentLayer)&&layerIsDet(prev)){
      opacity=prevProbOpacity;
      document.getElementById('opacity-slider').value=Math.round(prevProbOpacity*100);
      document.getElementById('opacity-val').textContent=Math.round(prevProbOpacity*100);
    }
    syncLegend(); drawColorbar(); refresh(); syncToggleBtns();
  });
  document.getElementById('opacity-slider').addEventListener('input',function(e){
    opacity=+e.target.value/100;
    document.getElementById('opacity-val').textContent=e.target.value; refresh();
  });
  document.getElementById('inund-slider').addEventListener('input',function(e){
    inundThresh=+e.target.value/100;
    document.getElementById('inund-val').textContent=e.target.value; refresh();
  });
  document.getElementById('depth-slider').addEventListener('input',function(e){
    depthThresh=+e.target.value;
    document.getElementById('depth-val').textContent=(+e.target.value).toFixed(2); refresh();
  });
  document.getElementById('coord-toggle').addEventListener('click',function(){
    showLatLon=!showLatLon;
    this.textContent=showLatLon?'Show row / col':'Show lat / lon';
    if(lastClickedCell)
      document.getElementById('exc-subtitle').textContent=
        cellLabel(lastClickedCell.r,lastClickedCell.c);
  });
  var infoOpen=true;
  document.getElementById('info-toggle').addEventListener('click',function(){
    infoOpen=!infoOpen;
    document.getElementById('info-content').style.display=infoOpen?'grid':'none';
    document.getElementById('tog-arrow').textContent=infoOpen?'▼':'▲';
  });

  // Discrete toggle
  document.getElementById('discrete-btn').addEventListener('click',function(){
    discreteMode=!discreteMode;
    this.className='btn'+(discreteMode?' btn-active':'');
    this.textContent=discreteMode?'▦ Discrete (on)':'▦ Discrete';
    drawColorbar(); refresh(); syncLegend();
  });

  var probBtn=document.getElementById('prob-btn');
  var detBtn =document.getElementById('det-btn');
  syncToggleBtns();
  if(probBtn&&detBtn){
    probBtn.addEventListener('click',function(){
      var prev=currentLayer;
      currentLayer='P(inundation)';
      document.getElementById('layer-select').value='P(inundation)';
      if(layerIsDet(prev)){ opacity=prevProbOpacity; document.getElementById('opacity-slider').value=Math.round(prevProbOpacity*100); document.getElementById('opacity-val').textContent=Math.round(prevProbOpacity*100); }
      syncLegend(); drawColorbar(); refresh(); syncToggleBtns();
    });
    detBtn.addEventListener('click',function(){
      var prev=currentLayer;
      currentLayer='Deterministic (m)';
      document.getElementById('layer-select').value='Deterministic (m)';
      if(!layerIsDet(prev)){ prevProbOpacity=opacity; opacity=1.0; document.getElementById('opacity-slider').value=100; document.getElementById('opacity-val').textContent='100'; }
      syncLegend(); drawColorbar(); refresh(); syncToggleBtns();
    });
  }

  document.getElementById('xs-yw-slider').addEventListener('input',function(){
    xsYWindow=parseFloat(this.value);
    document.getElementById('xs-yw-val').textContent=this.value;
    if(xsPts.length===2) drawXsProfile(xsPts[0],xsPts[1]);
  });

  if(XS_ENABLED){
    document.getElementById('xs-btn').addEventListener('click', toggleXsMode);
    document.getElementById('xs-save-btn').addEventListener('click',function(){
      var a=document.createElement('a');
      a.download='cross_section.png';
      a.href=document.getElementById('xs-canvas').toDataURL();
      a.click();
    });
    document.getElementById('xs-close-btn').addEventListener('click', toggleXsMode);
    document.getElementById('xs-redraw-btn').addEventListener('click',function(){
      if(xsPts.length===2) drawXsProfile(xsPts[0],xsPts[1]);
    });
  }

  document.getElementById('exc-redraw-btn').addEventListener('click',function(){
    if(lastClickedCell) drawExceedance(lastClickedCell.r, lastClickedCell.c);
  });

  makeDraggable('exc-panel','exc-drag-handle');
  makeDraggable('xs-panel','xs-drag-handle');

  requestAnimationFrame(function(){
    var p=document.getElementById('exc-panel');
    p.style.left=(window.innerWidth -p.offsetWidth -20)+'px';
    p.style.top =(window.innerHeight-p.offsetHeight-60)+'px';
  });
}

// ── Rendering ──────────────────────────────────────────────────────────────
function renderCanvas(){
  var lyr=LAYERS[currentLayer];
  var prob=LAYERS['P(inundation)'].data;
  var isDet=layerIsDet(currentLayer);
  var edges=bucketEdgesFor(currentLayer);
  var cmap=cmapFor(currentLayer);
  var imgData=offCtx.createImageData(NCOLS,NROWS), px=imgData.data;
  for(var r=0;r<NROWS;r++){
    for(var c=0;c<NCOLS;c++){
      var idx=(r*NCOLS+c)*4, v=lyr.data[r][c];
      var pI=isDet?(v>depthThresh?1:0):prob[r][c];
      if(v===null||(!isDet&&pI<inundThresh)||v<=depthThresh){px[idx+3]=0;continue;}
      var col;
      if(discreteMode){
        col=bucketColor(cmap,edges,v);
      } else {
        var t=Math.min(Math.max((v-lyr.vmin)/(lyr.vmax-lyr.vmin),0),1);
        col=lerpColor(cmap,t);
      }
      px[idx]=col[0]; px[idx+1]=col[1]; px[idx+2]=col[2];
      px[idx+3]=Math.round(opacity*255);
    }
  }
  offCtx.putImageData(imgData,0,0);
  return offCanvas.toDataURL();
}
function refresh(){overlay.setUrl(renderCanvas());}

// ── Colorbar ─────────────────────────────────────────────────────────────
function drawColorbar(){
  var lyr=LAYERS[currentLayer];
  if(!lyr) return;
  var cb=document.getElementById('cb-canvas');
  var ctx=cb.getContext('2d');
  var edges=bucketEdgesFor(currentLayer);
  var cmap=cmapFor(currentLayer);
  if(discreteMode){
    var n=edges.length-1;
    var bw=cb.width/n;
    for(var i=0;i<n;i++){
      var col=lerpColor(cmap,(i+0.5)/(n-1||1));
      ctx.fillStyle='rgb('+col[0]+','+col[1]+','+col[2]+')';
      ctx.fillRect(Math.round(i*bw),0,Math.ceil(bw),cb.height);
    }
  } else {
    for(var x=0;x<cb.width;x++){
      var c=lerpColor(cmap,x/(cb.width-1));
      ctx.fillStyle='rgb('+c[0]+','+c[1]+','+c[2]+')';
      ctx.fillRect(x,0,1,cb.height);
    }
  }
  document.getElementById('cb-title').textContent=currentLayer;
  document.getElementById('cb-min').textContent=lyr.vmin.toFixed(2)+(lyr.unit||'');
  document.getElementById('cb-max').textContent=lyr.vmax.toFixed(2)+(lyr.unit||'');
  var maxEl=document.getElementById('cb-maxval');
  var mx=LAYER_MAXES[currentLayer];
  maxEl.textContent=(mx!==null&&mx!==undefined)
    ?'Max for this layer: '+Number(mx).toFixed(2)+' m':'';
}

// ── Context-sensitive legend ──────────────────────────────────────────────
function updateLegendContent(){
  if(!window.legendDiv) return;
  var cmap=cmapFor(currentLayer);
  var edges=bucketEdgesFor(currentLayer);
  var isProb=(currentLayer==='P(inundation)');
  var isDepth=layerIsDepth(currentLayer);
  var isRange=layerIsRange(currentLayer);
  var n=edges.length-1;

  var html='<div class="leg-title">'+(isProb?'Flood likelihood':isRange?'Depth uncertainty':'Flood depth')+'</div>';

  if(discreteMode){
    // Show coloured bucket swatches
    for(var i=0;i<n;i++){
      var col=lerpColor(cmap,(i+0.5)/(n-1||1));
      var rgb='rgb('+col[0]+','+col[1]+','+col[2]+')';
      var lo=edges[i], hi=edges[i+1];
      var lbl;
      if(isProb)       lbl=(lo*100).toFixed(0)+'–'+(hi*100).toFixed(0)+'%';
      else             lbl=lo.toFixed(2)+'–'+hi.toFixed(2)+' m';
      if(i===n-1){
        if(isProb)     lbl='>'+(lo*100).toFixed(0)+'%';
        else           lbl='>'+lo.toFixed(2)+' m';
      }
      html+='<div class="leg-row"><span style="background:'+rgb+';outline:1px solid rgba(255,255,255,0.15)"></span>'+lbl+'</div>';
    }
  } else {
    // Continuous mode — descriptive bands
    if(isProb){
      html+=
        '<div class="leg-row"><span style="background:#800026"></span>High — &gt;70% chance</div>'+
        '<div class="leg-row"><span style="background:#fd8d3c"></span>Medium — 30–70%</div>'+
        '<div class="leg-row"><span style="background:#ffffcc;outline:1px solid #7a7a5a"></span>Low — &lt;30% chance</div>';
    } else if(isDepth){
      html+=
        '<div class="leg-row"><span style="background:#08306b"></span>Deep — &gt;1.0 m</div>'+
        '<div class="leg-row"><span style="background:#2171b5"></span>Moderate — 0.3–1.0 m</div>'+
        '<div class="leg-row"><span style="background:#c6dbef"></span>Shallow — &lt;0.3 m</div>';
    } else if(isRange){
      html+=
        '<div class="leg-row"><span style="background:#7a0177"></span>High sensitivity</div>'+
        '<div class="leg-row"><span style="background:#dd3497"></span>Moderate</div>'+
        '<div class="leg-row"><span style="background:#fee0d2"></span>Low sensitivity</div>';
    }
  }

  html+='<div class="leg-sep" style="margin-top:8px">Map features</div>'+
    '<div class="leg-row"><span style="background:none;border:2px dashed #f97316;height:0;margin-top:5px;flex-shrink:0"></span>Study area</div>';
  legendDiv.innerHTML=html;
}

function syncLegend(){
  if(!window.legendCtrl) return;
  var shouldShow=(currentLayer!=='' && LAYERS[currentLayer]!==undefined);
  if(shouldShow&&!legendOnMap){legendCtrl.addTo(_map);legendOnMap=true;}
  else if(!shouldShow&&legendOnMap){legendCtrl.remove();legendOnMap=false;}
  updateLegendContent();
}

// ── Helpers ──────────────────────────────────────────────────────────────
function cellToLatLon(r,c){
  return {
    lat: LAT_MAX - r/(NROWS-1)*(LAT_MAX-LAT_MIN),
    lon: LON_MIN + c/(NCOLS-1)*(LON_MAX-LON_MIN)
  };
}
function cellLabel(r,c){
  if(showLatLon){
    var ll=cellToLatLon(r,c);
    return 'Lat\u00a0'+ll.lat.toFixed(5)+',\u00a0Lon\u00a0'+ll.lon.toFixed(5);
  }
  return 'Row\u00a0'+r+',\u00a0Col\u00a0'+c;
}

// ── Depth exceedance chart ────────────────────────────────────────────────
// Axes: x = P(exceedance), y = depth  (flipped from original)
function drawExceedance(r,c){
  document.getElementById('exc-prompt').style.display='none';
  var canvas=document.getElementById('exc-canvas');
  canvas.style.display='block';
  canvas.width =canvas.offsetWidth  || 310;
  canvas.height=canvas.offsetHeight || 320;
  document.getElementById('exc-subtitle').textContent=cellLabel(r,c);

  var depths=[], excProbs=[];
  for(var i=0;i<PROB_LEVELS.length;i++){
    depths.push(QUANT_DEPTHS[i][r][c]);
    excProbs.push(1-PROB_LEVELS[i]);
  }

  var ctx=canvas.getContext('2d'), W=canvas.width, H=canvas.height;
  // x = exceedance prob (0→1), y = depth (0→yMax) — depth increases upward
  var pad={t:16,r:16,b:50,l:52}, pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#0a1220'; ctx.fillRect(0,0,W,H);

  var yMax=Math.max(depths[depths.length-1]*1.1,0.01);
  var yMin=Math.max(0, depths[0]-0.10);   // 10 cm below P05, clamped to 0
  if(depths[Math.floor(depths.length/2)]<0.001){
    ctx.fillStyle='#475569'; ctx.font='14px Segoe UI,Arial'; ctx.textAlign='center';
    ctx.fillText('No significant flooding at this cell',W/2,H/2);
    return;
  }

  // Helper: canvas coords from (excProb, depth)
  function px(ep){ return pad.l + ep*pw; }
  function py(d){  return pad.t + ph*(1 - (d-yMin)/(yMax-yMin)); }

  // gridlines
  ctx.strokeStyle='#1e293b'; ctx.lineWidth=0.5;
  [0,0.2,0.4,0.6,0.8,1.0].forEach(function(f){
    ctx.beginPath(); ctx.moveTo(px(f),pad.t); ctx.lineTo(px(f),pad.t+ph); ctx.stroke();
  });
  [0,0.25,0.5,0.75,1.0].forEach(function(f){
    var dv=yMin+f*(yMax-yMin);
    ctx.beginPath(); ctx.moveTo(pad.l,py(dv)); ctx.lineTo(pad.l+pw,py(dv)); ctx.stroke();
  });
  // axes
  ctx.strokeStyle='#334155'; ctx.lineWidth=1;
  ctx.beginPath();
  ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ph); ctx.lineTo(pad.l+pw,pad.t+ph);
  ctx.stroke();

  // filled area under curve
  ctx.beginPath();
  ctx.moveTo(px(excProbs[0]), py(depths[0]));
  for(var i=1;i<depths.length;i++)
    ctx.lineTo(px(excProbs[i]), py(depths[i]));
  ctx.lineTo(px(excProbs[depths.length-1]), py(yMin));
  ctx.lineTo(px(excProbs[0]), py(yMin));
  ctx.closePath();
  ctx.fillStyle='rgba(59,130,246,0.12)'; ctx.fill();

  // exceedance curve
  ctx.strokeStyle='#3b82f6'; ctx.lineWidth=2.5; ctx.beginPath();
  for(var i=0;i<depths.length;i++){
    i===0?ctx.moveTo(px(excProbs[i]),py(depths[i])):ctx.lineTo(px(excProbs[i]),py(depths[i]));
  }
  ctx.stroke();

  // deterministic horizontal line
  if(LAYERS['Deterministic (m)']){
    var vdet=LAYERS['Deterministic (m)'].data[r][c];
    if(vdet>0.001){
      var ydet=py(vdet);
      ctx.strokeStyle='#f97316'; ctx.lineWidth=2.5; ctx.setLineDash([5,3]);
      ctx.beginPath(); ctx.moveTo(pad.l,ydet); ctx.lineTo(pad.l+pw,ydet); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#f97316'; ctx.font='bold 11px Segoe UI,Arial'; ctx.textAlign='left';
      ctx.fillText('Det. '+vdet.toFixed(2)+'m', pad.l+4, ydet-4);
    }
  }

  // percentile markers — horizontal lines at key depths
  var markers=[
    {p:0.05, col:'#f87171', lbl:'P05'},
    {p:0.50, col:'#34d399', lbl:'P50'},
    {p:0.95, col:'#fbbf24', lbl:'P95'}
  ];
  markers.forEach(function(m){
    var pi=PROB_LEVELS.indexOf(m.p);
    if(pi<0) return;
    m.d  = depths[pi];
    m.ep = excProbs[pi];
    var yd=py(m.d), xp=px(m.ep);
    // dashed horizontal at this depth
    ctx.strokeStyle=m.col; ctx.lineWidth=0.8; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(pad.l,yd); ctx.lineTo(xp,yd); ctx.stroke();
    // dashed vertical at this exceedance prob
    ctx.beginPath(); ctx.moveTo(xp,yd); ctx.lineTo(xp,pad.t+ph); ctx.stroke();
    ctx.setLineDash([]);
    // dot on curve
    ctx.fillStyle=m.col;
    ctx.beginPath(); ctx.arc(xp,yd,3.5,0,2*Math.PI); ctx.fill();
    // label on y-axis
    ctx.fillStyle=m.col; ctx.font='bold 11px Segoe UI,Arial'; ctx.textAlign='right';
    ctx.fillText(m.lbl+' '+m.d.toFixed(2)+'m', pad.l-4, yd+4);
  });

  // x-axis labels (exceedance probability)
  ctx.fillStyle='#94a3b8'; ctx.font='12px Segoe UI,Arial';
  [0,0.2,0.4,0.6,0.8,1.0].forEach(function(f){
    ctx.textAlign='center';
    ctx.fillText((f).toFixed(1), px(f), pad.t+ph+13);
  });
  // y-axis labels (depth)
  [0,0.25,0.5,0.75,1.0].forEach(function(f){
    var dv=yMin+f*(yMax-yMin);
    ctx.textAlign='right';
    ctx.fillText(dv.toFixed(2), pad.l-4, py(dv)+4);
  });

  // axis titles
  ctx.fillStyle='#cbd5e1'; ctx.font='13px Segoe UI,Arial'; ctx.textAlign='center';
  ctx.fillText('P(depth exceeded)', pad.l+pw/2, H-3);
  ctx.save(); ctx.translate(13, pad.t+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Depth (m)', 0, 0); ctx.restore();

  ctx.fillStyle='#334155'; ctx.font='11px Segoe UI,Arial'; ctx.textAlign='right';
  ctx.fillText('N='+PROB_LEVELS.length+' levels', W-4, pad.t+10);
}

// ── Cross-section ──────────────────────────────────────────────────────────
function latLonToRowCol(lat,lon){
  var r=Math.min(Math.max(Math.round((LAT_MAX-lat)/(LAT_MAX-LAT_MIN)*(NROWS-1)),0),NROWS-1);
  var c=Math.min(Math.max(Math.round((lon-LON_MIN)/(LON_MAX-LON_MIN)*(NCOLS-1)),0),NCOLS-1);
  var x=XLL+c*CS_M, y=YLL+(NROWS-1-r)*CS_M;
  return {r:r,c:c,x:x,y:y};
}
function sampleAlongLine(key,r0,c0,r1,c1,n){
  var vals=[];
  for(var i=0;i<n;i++){
    var t=i/(n-1);
    var r=Math.round(r0+t*(r1-r0)), c=Math.round(c0+t*(c1-c0));
    if(r>=0&&r<NROWS&&c>=0&&c<NCOLS){
      var v=XS_DATA[key][r][c];
      vals.push(v===null?NaN:v);
    } else vals.push(NaN);
  }
  return vals;
}

function drawXsProfile(p0,p1){
  var dx=p1.x-p0.x, dy=p1.y-p0.y;
  var lenM=Math.sqrt(dx*dx+dy*dy);
  var n=Math.max(Math.ceil(lenM/CS_M),2);

  var dem   =sampleAlongLine('dem',   p0.r,p0.c,p1.r,p1.c,n);
  var p05   =sampleAlongLine('p05',   p0.r,p0.c,p1.r,p1.c,n);
  var p50   =sampleAlongLine('p50',   p0.r,p0.c,p1.r,p1.c,n);
  var p95   =sampleAlongLine('p95',   p0.r,p0.c,p1.r,p1.c,n);
  var det   =sampleAlongLine('det',   p0.r,p0.c,p1.r,p1.c,n);
  var det_cc=sampleAlongLine('det_cc',p0.r,p0.c,p1.r,p1.c,n);

  function wse(demVal,depth){
    if(isNaN(demVal)) return NaN;
    if(!depth||depth<=0) return NaN;
    return demVal+depth;
  }

  var p05w=dem.map(function(d,i){return wse(d,p05[i]);});
  var p50w=dem.map(function(d,i){return wse(d,p50[i]);});
  var p95w=dem.map(function(d,i){return wse(d,p95[i]);});
  var detw=dem.map(function(d,i){return wse(d,det[i]);});
  var dccw=dem.map(function(d,i){return wse(d,det_cc[i]);});

  var validDem=dem.filter(function(v){return !isNaN(v);});
  var meanElev=validDem.reduce(function(a,b){return a+b;},0)/validDem.length;
  var yw=xsYWindow;
  var yMin=meanElev-yw, yMax=meanElev+yw;

  var canvas=document.getElementById('xs-canvas');
  canvas.width=canvas.offsetWidth||500;
  canvas.height=canvas.offsetHeight||300;
  var ctx=canvas.getContext('2d');
  var W=canvas.width, H=canvas.height;
  var pad={t:28,r:20,b:52,l:58}, pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;

  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#0a1220'; ctx.fillRect(0,0,W,H);

  function pxX(i){return pad.l+(i/(n-1))*pw;}
  function pxY(v){return pad.t+(1-(v-yMin)/(yMax-yMin))*ph;}
  function clampY(v){return Math.max(pad.t,Math.min(pad.t+ph,pxY(v)));}

  ctx.strokeStyle='#1e293b'; ctx.lineWidth=0.5;
  for(var gi=0;gi<=4;gi++){
    var gv=yMin+gi/4*(yMax-yMin), gyp=pxY(gv);
    ctx.beginPath(); ctx.moveTo(pad.l,gyp); ctx.lineTo(pad.l+pw,gyp); ctx.stroke();
  }

  // DEM fill
  ctx.beginPath();
  var fs=false;
  for(var i=0;i<n;i++){
    if(isNaN(dem[i])) continue;
    if(!fs){ctx.moveTo(pxX(i),pad.t+ph); ctx.lineTo(pxX(i),clampY(dem[i])); fs=true;}
    else ctx.lineTo(pxX(i),clampY(dem[i]));
  }
  ctx.lineTo(pxX(n-1),pad.t+ph); ctx.closePath();
  ctx.fillStyle='rgba(160,82,45,0.30)'; ctx.fill();

  // p05–p95 envelope
  ctx.beginPath(); fs=false;
  for(var i=0;i<n;i++){
    var top=isNaN(p95w[i])?(isNaN(dem[i])?NaN:dem[i]):Math.max(p95w[i],dem[i]);
    if(isNaN(top)) continue;
    if(!fs){ctx.moveTo(pxX(i),clampY(top)); fs=true;}
    else ctx.lineTo(pxX(i),clampY(top));
  }
  for(var i=n-1;i>=0;i--){
    var bot=isNaN(p05w[i])?(isNaN(dem[i])?NaN:dem[i]):Math.max(p05w[i],dem[i]);
    if(!isNaN(bot)) ctx.lineTo(pxX(i),clampY(bot));
  }
  ctx.closePath();
  ctx.fillStyle='rgba(59,130,246,0.18)'; ctx.fill();

  function drawLine(vals,color,width,dash){
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.setLineDash(dash||[]);
    ctx.beginPath(); var s=false;
    for(var i=0;i<n;i++){
      if(isNaN(vals[i])){s=false;continue;}
      var yp=clampY(vals[i]);
      if(!s){ctx.moveTo(pxX(i),yp);s=true;}else ctx.lineTo(pxX(i),yp);
    }
    ctx.stroke(); ctx.setLineDash([]);
  }

  drawLine(dem,'#a0522d',1.5);
  var p50_plot=p50w.map(function(v,i){return(isNaN(v)||isNaN(dem[i]))?NaN:Math.max(v,dem[i]);});
  var det_plot=detw.map(function(v,i){return(isNaN(v)||isNaN(dem[i]))?NaN:Math.max(v,dem[i]);});
  var dcc_plot=dccw.map(function(v,i){return(isNaN(v)||isNaN(dem[i]))?NaN:Math.max(v,dem[i]);});

  drawLine(p50_plot,'#3b82f6',2.0);
  drawLine(det_plot,'#e2e8f0',2.5);
  drawLine(dcc_plot,'#ef4444',2.0,[6,3]);

  ctx.strokeStyle='#334155'; ctx.lineWidth=1;
  ctx.beginPath();
  ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ph); ctx.lineTo(pad.l+pw,pad.t+ph);
  ctx.stroke();

  ctx.fillStyle='#94a3b8'; ctx.font='14px Segoe UI,Arial'; ctx.textAlign='right';
  for(var gi=0;gi<=4;gi++){
    var gv=yMin+gi/4*(yMax-yMin);
    ctx.fillText(gv.toFixed(1),pad.l-5,pxY(gv)+4);
  }
  ctx.textAlign='center';
  for(var xi=0;xi<=4;xi++){
    ctx.fillText((xi/4*lenM).toFixed(0)+'m',pad.l+xi/4*pw,pad.t+ph+14);
  }
  ctx.fillStyle='#cbd5e1'; ctx.font='14px Segoe UI,Arial'; ctx.textAlign='center';
  ctx.fillText('Distance (m)',pad.l+pw/2,H-4);
  ctx.save(); ctx.translate(14,pad.t+ph/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Elevation (m AOD)',0,0); ctx.restore();

  ctx.fillStyle='#f1f5f9'; ctx.font='14px Segoe UI,Arial'; ctx.textAlign='left';
  ctx.fillText(lenM.toFixed(0)+'m  |  angle: '+Math.round(Math.atan2(dy,dx)*180/Math.PI)+'°',pad.l,18);

  // Human figure
  var humanHeight=1.8, humanX=pad.l+pw*0.97;
  var humanColIdx=n-1;
  var humanGroundElev=dem[humanColIdx];
  if(isNaN(humanGroundElev)) humanGroundElev=meanElev;
  if(humanGroundElev>=yMin&&humanGroundElev<=yMax){
    var hBase=clampY(humanGroundElev), hTop=clampY(humanGroundElev+humanHeight);
    var hPx=hBase-hTop;
    if(hPx>4){
      ctx.strokeStyle='#f59e0b'; ctx.fillStyle='#f59e0b'; ctx.lineWidth=1.5;
      var headR=Math.max(hPx*0.13,3);
      ctx.beginPath(); ctx.arc(humanX,hTop+headR,headR,0,2*Math.PI); ctx.fill();
      var neckY=hTop+headR*2, waistY=hTop+hPx*0.55, armY=hTop+hPx*0.38;
      ctx.beginPath(); ctx.moveTo(humanX,neckY); ctx.lineTo(humanX,waistY); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(humanX-hPx*0.18,armY); ctx.lineTo(humanX+hPx*0.18,armY); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(humanX,waistY); ctx.lineTo(humanX-hPx*0.14,hBase); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(humanX,waistY); ctx.lineTo(humanX+hPx*0.14,hBase); ctx.stroke();
      ctx.fillStyle='#f59e0b'; ctx.font='10px Segoe UI,Arial'; ctx.textAlign='center';
      ctx.fillText('1.8m',humanX,hBase+12);
    }
  }

  var leg=[['#a0522d','DEM',[]],['#3b82f6','Emulator p50',[]],
           ['rgba(59,130,246,0.35)','p05–p95',[]],['#e2e8f0','Deterministic',[]],['#ef4444','Det.+CC',[5,3]]];
  var lx=pad.l, ly2=pad.t+ph+33;
  ctx.font='12px Segoe UI,Arial';
  leg.forEach(function(l){
    ctx.strokeStyle=l[0]; ctx.lineWidth=2; ctx.setLineDash(l[2]);
    ctx.beginPath(); ctx.moveTo(lx,ly2); ctx.lineTo(lx+16,ly2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#94a3b8'; ctx.textAlign='left';
    ctx.fillText(l[1],lx+20,ly2+4);
    lx+=ctx.measureText(l[1]).width+36;
  });

  document.getElementById('xs-status').textContent=
    'Length: '+lenM.toFixed(0)+'m  \u00b7  Click START to draw new line';
}

function toggleXsMode(){
  if(!XS_ENABLED) return;
  xsMode=!xsMode;
  var btn=document.getElementById('xs-btn');
  var panel=document.getElementById('xs-panel');
  var excPanel=document.getElementById('exc-panel');
  if(xsMode){
    btn.style.background='#1e3a5f'; btn.style.color='#93c5fd';
    panel.style.display='block';
    if(!panel.style.top||panel.style.top===''){
      panel.style.left=Math.round((window.innerWidth-panel.offsetWidth)/2)+'px';
      panel.style.top=(window.innerHeight-panel.offsetHeight-20)+'px';
    }
    excPanel.style.display='none';
    document.getElementById('xs-status').textContent='Click START point on the map';
    _map.getContainer().style.cursor='crosshair';
    xsPts=[]; xsMarkers.forEach(function(m){_map.removeLayer(m);}); xsMarkers=[];
    if(xsLine){_map.removeLayer(xsLine); xsLine=null;}
  } else {
    btn.style.background=''; btn.style.color='';
    panel.style.display='none';
    excPanel.style.display='block';
    _map.getContainer().style.cursor='';
    xsPts=[]; xsMarkers.forEach(function(m){_map.removeLayer(m);}); xsMarkers=[];
    if(xsLine){_map.removeLayer(xsLine); xsLine=null;}
  }
}

function syncToggleBtns(){
  var probBtn=document.getElementById('prob-btn');
  var detBtn =document.getElementById('det-btn');
  if(!probBtn||!detBtn) return;
  var isDet=layerIsDet(currentLayer);
  probBtn.className='btn'+(isDet?'':' btn-active');
  detBtn.className ='btn'+(isDet?' btn-active':'');
}

function makeDraggable(panelId,handleId){
  var panel=document.getElementById(panelId);
  var handle=document.getElementById(handleId);
  if(!panel||!handle) return;
  var dragging=false,startX,startY,origLeft,origTop;
  handle.addEventListener('mousedown',function(e){
    if(e.target.closest&&e.target.closest('button')) return;
    dragging=true;
    startX=e.clientX; startY=e.clientY;
    origLeft=parseInt(panel.style.left)||panel.getBoundingClientRect().left;
    origTop =parseInt(panel.style.top) ||panel.getBoundingClientRect().top;
    handle.style.cursor='grabbing'; e.preventDefault();
  });
  document.addEventListener('mousemove',function(e){
    if(!dragging) return;
    panel.style.left=Math.min(Math.max(origLeft+(e.clientX-startX),0),window.innerWidth -panel.offsetWidth)+'px';
    panel.style.top =Math.min(Math.max(origTop +(e.clientY-startY),0),window.innerHeight-panel.offsetHeight)+'px';
  });
  document.addEventListener('mouseup',function(){
    if(!dragging) return; dragging=false; handle.style.cursor='grab';
  });
}
"""


# ---------------------------------------------------------------------------
# Info cards
# ---------------------------------------------------------------------------
def _build_info_cards(return_period, n_mc, det_card):
    return f"""
<div class="icard">
  <div class="icard-head">Probability of inundation</div>
  <p>The chance a location floods during a 1-in-{return_period} yr event, from {n_mc:,} simulations
  sampling uncertainty in inflow magnitude and timing. 0.9 = flooded in 9 of 10 runs.
  Pale yellow = low risk; deep red = high risk (&gt;70%).</p>
</div>
<div class="icard">
  <div class="icard-head">Median depth (P50)</div>
  <p>The 50th-percentile flood depth across {n_mc:,} simulations. Half predicted shallower,
  half deeper. Best single-number summary for general flood-risk communication.</p>
</div>
<div class="icard">
  <div class="icard-head">P05 — optimistic scenario</div>
  <p>Near best-case: only 5% of runs predicted shallower flooding. Illustrates the
  lower-bound plausible impact. Not suitable for planning or design on its own.</p>
</div>
<div class="icard">
  <div class="icard-head">P95 — precautionary scenario</div>
  <p>Near worst-case: only 5% of runs predicted deeper flooding. Use for critical
  infrastructure, vulnerable properties, insurance, or emergency planning.</p>
</div>
<div class="icard">
  <div class="icard-head">Sensitivity range maps</div>
  <p><b>P95−Median</b> and <b>P95−P05</b> show where predictions are most uncertain.
  High values indicate hydrologically sensitive areas where more freeboard may be
  warranted in design. Use alongside P95 for robust planning decisions.</p>
</div>
<div class="icard">
  <div class="icard-head">Depth exceedance curve</div>
  <p>Click any cell to see the full depth-uncertainty distribution at that point.
  Shows P(depth &gt; d) vs d — read off: "there is a X% chance depth exceeds Y m".
  Orange dashed line shows the deterministic baseline depth.</p>
</div>
{det_card}"""


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------
def _build_html(site_name, return_period, n_mc, overlay_alpha,
                xs_btn_html, det_option, det_toggle_btn,
                n_info_cols, info_cards, data_js, logic_js,
                data_loader_js, embedded_data_tags, about_page_url,
                leaflet_css, leaflet_js):

    range_options = """
        <option>P95&#8722;Median (m)</option>
        <option>P95&#8722;P05 (m)</option>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{site_name} — 1-in-{return_period} yr flood explorer</title>
{leaflet_css}
{leaflet_js}
<style>
:root{{
  --fs-xs:  clamp(11px, 1.2vw, 15px);
  --fs-sm:  clamp(13px, 1.4vw, 18px);
  --fs-md:  clamp(14px, 1.6vw, 21px);
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',Arial,sans-serif;font-size:var(--fs-sm);background:#0f172a;
     color:#e2e8f0;display:flex;flex-direction:column;height:100vh;overflow:hidden;}}
#hdr{{padding:9px 18px;background:#1e293b;border-bottom:1px solid #334155;
      display:flex;align-items:center;justify-content:space-between;flex-shrink:0;}}
#hdr h1{{font-size:var(--fs-md);font-weight:600;color:#f1f5f9;}}
#hdr span{{font-size:var(--fs-xs);color:#64748b;}}
#map-wrap{{flex:1;position:relative;min-height:0;}}
#map{{width:100%;height:100%;}}
/* Loading overlay */
#loading-overlay{{position:absolute;inset:0;background:rgba(10,18,32,0.88);
  z-index:9999;display:flex;flex-direction:column;align-items:center;justify-content:center;}}
#loading-overlay p{{color:#60a5fa;font-size:16px;margin-top:16px;}}
.spinner{{width:48px;height:48px;border:4px solid #1e3a5f;
  border-top-color:#3b82f6;border-radius:50%;animation:spin 0.9s linear infinite;}}
@keyframes spin{{to{{transform:rotate(360deg);}}}}
.panel{{position:absolute;z-index:1000;background:rgba(10,18,35,0.93);
        border:1px solid #1e3a5f;border-radius:9px;backdrop-filter:blur(8px);
        overflow:hidden;}}
.panel-head{{font-size:var(--fs-xs);text-transform:uppercase;letter-spacing:.08em;
             color:#4a7fb5;margin-bottom:10px;font-weight:600;}}
#ctrl-panel{{top:10px;right:10px;padding:14px 16px;width:275px;}}
.ctrl{{margin-bottom:10px;}}
.ctrl label{{display:block;color:#94a3b8;font-size:var(--fs-xs);margin-bottom:4px;}}
.ctrl label b{{color:#cbd5e1;font-weight:600;}}
.ctrl select{{width:100%;background:#0f2236;color:#e2e8f0;border:1px solid #1e3a5f;
              border-radius:5px;padding:6px 9px;font-size:var(--fs-sm);cursor:pointer;appearance:none;}}
.ctrl input[type=range]{{width:100%;accent-color:#3b82f6;cursor:pointer;margin-top:2px;}}
.sep{{border:none;border-top:1px solid #1a2a3a;margin:10px 0;}}
.ctrl-note{{font-size:var(--fs-xs);color:#334155;margin-top:3px;}}
.btn{{width:100%;margin-top:4px;padding:6px 9px;background:#0f2236;color:#60a5fa;
      border:1px solid #1e3a5f;border-radius:5px;font-size:var(--fs-xs);cursor:pointer;
      transition:background 0.15s;}}
.btn:hover{{background:#162032;}}
.btn-active{{background:#1e3a5f !important;color:#93c5fd !important;border-color:#3b82f6 !important;}}
.about-btn{{display:inline-block;padding:4px 12px;margin-left:12px;
  background:#0f2236;color:#60a5fa;border:1px solid #1e3a5f;border-radius:5px;
  font-size:var(--fs-xs);cursor:pointer;text-decoration:none;}}
.about-btn:hover{{background:#162032;}}
#cb-panel{{top:10px;left:10px;padding:10px 14px;width:190px;}}
#cb-canvas{{width:100%;height:12px;border-radius:3px;display:block;}}
#cb-labels{{display:flex;justify-content:space-between;font-size:var(--fs-xs);color:#4a7fb5;margin-top:4px;}}
#cb-maxval{{font-size:var(--fs-xs);color:#64748b;margin-top:5px;text-align:right;
            padding-top:4px;border-top:1px solid #1e3a5f;}}
.flood-legend{{background:rgba(10,18,35,0.93)!important;border:1px solid #1e3a5f!important;
               border-radius:9px;padding:11px 14px!important;margin-top:100px!important;}}
.leg-title{{font-size:var(--fs-xs);text-transform:uppercase;letter-spacing:.08em;color:#4a7fb5;
            font-weight:600;margin-bottom:8px;}}
.leg-sep{{font-size:var(--fs-xs);color:#334155;margin:6px 0 5px;text-transform:uppercase;letter-spacing:.05em;}}
.leg-row{{display:flex;align-items:center;gap:8px;font-size:var(--fs-sm);color:#94a3b8;margin-bottom:5px;}}
.leg-row span{{width:20px;height:11px;border-radius:2px;flex-shrink:0;display:inline-block;}}
#exc-panel{{top:10px;left:10px;padding:0;width:330px;user-select:none;
            resize:both;overflow:auto;min-width:240px;min-height:220px;}}
#exc-drag-handle{{cursor:grab;display:flex;justify-content:space-between;align-items:center;
  padding:10px 13px 8px 13px;border-bottom:1px solid #1e3a5f;font-size:var(--fs-xs);
  text-transform:uppercase;letter-spacing:.08em;color:#4a7fb5;font-weight:600;
  border-radius:9px 9px 0 0;}}
#exc-drag-handle:active{{cursor:grabbing;}}
#exc-drag-handle:hover{{background:rgba(30,58,95,0.4);}}
#exc-body{{padding:8px 13px 12px 13px;}}
#exc-subtitle{{font-size:var(--fs-xs);color:#4a7fb5;margin-bottom:7px;min-height:14px;}}
#exc-prompt{{font-size:var(--fs-sm);color:#334155;text-align:center;padding:26px 0;}}
#exc-canvas{{display:none;width:100%;border-radius:3px;}}
#xs-panel{{display:none;top:10px;left:10px;padding:0;width:640px;user-select:none;
           resize:both;overflow:auto;min-width:300px;min-height:200px;}}
#xs-drag-handle{{cursor:grab;display:flex;justify-content:space-between;align-items:center;
  padding:10px 14px 8px 14px;border-bottom:1px solid #1e3a5f;font-size:var(--fs-xs);
  text-transform:uppercase;letter-spacing:.08em;color:#4a7fb5;font-weight:600;
  border-radius:9px 9px 0 0;}}
#xs-drag-handle:active{{cursor:grabbing;}}
#xs-drag-handle:hover{{background:rgba(30,58,95,0.4);}}
#xs-body{{padding:12px 14px;}}
#xs-status{{font-size:var(--fs-xs);color:#4a7fb5;margin-bottom:6px;min-height:16px;}}
#xs-canvas{{width:100%;height:100%;min-height:220px;border-radius:4px;display:block;}}
#xs-controls{{display:flex;align-items:center;gap:10px;margin-top:8px;font-size:var(--fs-xs);color:#94a3b8;}}
#xs-controls input[type=range]{{accent-color:#3b82f6;width:90px;}}
#info-panel{{background:#111d2e;border-top:1px solid #1e3a5f;flex-shrink:0;}}
#info-toggle{{width:100%;padding:7px 18px;background:#111d2e;color:#4a7fb5;border:none;
              cursor:pointer;text-align:left;font-size:var(--fs-sm);display:flex;align-items:center;gap:7px;}}
#info-toggle:hover{{background:#162032;color:#93c5fd;}}
#info-content{{display:grid;grid-template-columns:repeat({n_info_cols},1fr);
               gap:10px;padding:10px 18px 14px;}}
.icard{{background:#0a1220;border:1px solid #1e3a5f;border-radius:7px;padding:10px 12px;}}
.icard-head{{font-size:var(--fs-sm);font-weight:600;color:#60a5fa;margin-bottom:5px;}}
.icard p{{font-size:var(--fs-xs);color:#64748b;line-height:1.55;}}
.leaflet-popup-content-wrapper{{background:rgba(10,18,35,0.97)!important;color:#e2e8f0!important;
  border:1px solid #1e3a5f!important;border-radius:9px!important;font-size:var(--fs-sm)!important;
  box-shadow:none!important;}}
.leaflet-popup-tip{{background:#111d2e!important;}}
.pr{{display:flex;justify-content:space-between;gap:14px;padding:2px 0;}}
.pk{{color:#64748b;}}.pv{{font-weight:600;color:#60a5fa;}}
</style>
</head>
<body>

<div id="loading-overlay">
  <div class="spinner"></div>
  <p>Loading flood data…</p>
</div>

<div id="hdr">
  <h1>{site_name} — probabilistic flood explorer</h1>
  <div style="display:flex;align-items:center;gap:8px;">
    <span>1-in-{return_period} yr &nbsp;·&nbsp; N={n_mc:,} Monte Carlo samples
          &nbsp;·&nbsp; Click map to analyse a point</span>
    {xs_btn_html}
    <a href="{about_page_url}" class="about-btn" title="About the emulator methodology">ℹ About</a>
  </div>
</div>

<div id="map-wrap">
  <div id="map"></div>

  <div class="panel" id="ctrl-panel">
    <div class="panel-head">Display controls</div>
    <div class="ctrl">
      <label>Layer</label>
      <select id="layer-select">
        {det_option}
        <option selected>P(inundation)</option>
        <option>Median depth (m)</option>
        <option>P05 depth (m)</option>
        <option>P95 depth (m)</option>
        {range_options}
      </select>
    </div>
    <div class="ctrl">
      <label>Opacity: <b><span id="opacity-val">{int(overlay_alpha*100)}</span>%</b></label>
      <input type="range" id="opacity-slider" min="10" max="100"
             value="{int(overlay_alpha*100)}" step="5">
    </div>
    <hr class="sep">
    <div class="ctrl">
      <label>Show cells flooded ≥ <b><span id="inund-val">0</span>%</b> of runs</label>
      <input type="range" id="inund-slider" min="0" max="99" value="0" step="1">
      <div class="ctrl-note">Raise to isolate high-confidence flood zones</div>
    </div>
    <div class="ctrl">
      <label>Hide depths below <b><span id="depth-val">0.05</span> m</b></label>
      <input type="range" id="depth-slider" min="0" max="0.5" value="0.05" step="0.01">
    </div>
    <hr class="sep">
    <button class="btn" id="coord-toggle">Show lat / lon</button>
    <button class="btn" id="discrete-btn">&#9638; Discrete</button>
    {det_toggle_btn}
  </div>

  <div class="panel" id="cb-panel">
    <div class="panel-head" id="cb-title">P(inundation)</div>
    <canvas id="cb-canvas" width="165" height="12"></canvas>
    <div id="cb-labels"><span id="cb-min">0</span><span id="cb-max">1</span></div>
    <div id="cb-maxval"></div>
  </div>

  <div class="panel" id="exc-panel">
    <div id="exc-drag-handle">
      Depth exceedance curve
      <span style="font-weight:400;letter-spacing:0;color:#334155;">⠿ drag</span>
    </div>
    <div id="exc-body">
      <div id="exc-subtitle"></div>
      <button id="exc-redraw-btn" style="padding:3px 10px;background:#0f2236;
        color:#60a5fa;border:1px solid #1e3a5f;border-radius:5px;
        font-size:12px;cursor:pointer;margin-bottom:6px;">↺ Redraw</button>
      <div id="exc-prompt">Click the flood map to analyse a point</div>
      <canvas id="exc-canvas" width="298" height="320"></canvas>
    </div>
  </div>

  <div class="panel" id="xs-panel">
    <div id="xs-drag-handle">
      ✂ Cross-section
      <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-weight:400;letter-spacing:0;color:#334155;">⠿ drag</span>
        <button id="xs-close-btn" style="background:none;border:none;color:#4a7fb5;
                cursor:pointer;font-size:16px;line-height:1;" title="Close">✕</button>
      </div>
    </div>
    <div id="xs-body">
      <div id="xs-status">Click START point on the map</div>
      <canvas id="xs-canvas"></canvas>
      <div id="xs-controls">
        <span>±Y window:</span>
        <input type="range" id="xs-yw-slider" min="1" max="20" value="5" step="0.5">
        <span id="xs-yw-val">5</span>m
        <button id="xs-redraw-btn" style="padding:4px 12px;background:#0f2236;
            color:#60a5fa;border:1px solid #1e3a5f;border-radius:5px;font-size:14px;cursor:pointer;">
            ↺ Redraw</button>
        <button id="xs-save-btn" style="margin-left:auto;padding:4px 12px;background:#0f2236;
                color:#60a5fa;border:1px solid #1e3a5f;border-radius:5px;font-size:14px;cursor:pointer;">
                Save PNG</button>
      </div>
    </div>
  </div>

</div><!-- map-wrap -->

<div id="info-panel">
  <button id="info-toggle"><span id="tog-arrow">▼</span> About these maps</button>
  <div id="info-content">{info_cards}</div>
</div>

<script>{data_js}</script>
{embedded_data_tags}
<script>{data_loader_js}</script>
<script>{logic_js}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# About page builder
# ---------------------------------------------------------------------------
def build_about_page(
    site_name="Inverurie",
    out_path="about.html",
    main_page_url="index.html",
):
    """
    Generate a standalone 'About the emulator' companion page.
    Populate the placeholder sections with project-specific content.
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{site_name} — About the flood emulator</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0;
      min-height:100vh;}}
header{{background:#1e293b;border-bottom:1px solid #334155;
        padding:14px 40px;display:flex;align-items:center;justify-content:space-between;}}
header h1{{font-size:clamp(15px,2vw,22px);font-weight:600;color:#f1f5f9;}}
.back-btn{{padding:5px 14px;background:#0f2236;color:#60a5fa;border:1px solid #1e3a5f;
           border-radius:5px;font-size:13px;cursor:pointer;text-decoration:none;}}
.back-btn:hover{{background:#162032;}}
main{{max-width:980px;margin:40px auto;padding:0 24px 60px;}}
.section{{background:#111d2e;border:1px solid #1e3a5f;border-radius:10px;
           padding:28px 32px;margin-bottom:28px;}}
.section h2{{font-size:clamp(14px,1.6vw,20px);font-weight:600;color:#60a5fa;margin-bottom:14px;}}
.section p{{font-size:clamp(12px,1.2vw,15px);color:#94a3b8;line-height:1.70;margin-bottom:10px;}}
.section ul{{padding-left:20px;}}
.section li{{font-size:clamp(12px,1.2vw,15px);color:#94a3b8;line-height:1.70;margin-bottom:6px;}}
.placeholder{{background:#0a1220;border:1px dashed #334155;border-radius:6px;
               padding:16px 20px;margin-top:12px;}}
.placeholder p{{color:#475569;font-style:italic;}}
.tag{{display:inline-block;padding:2px 9px;border-radius:12px;font-size:11px;
      font-weight:600;margin-right:5px;margin-bottom:4px;background:#1e3a5f;color:#93c5fd;}}
footer{{text-align:center;padding:20px;color:#334155;font-size:12px;border-top:1px solid #1e293b;}}
</style>
</head>
<body>
<header>
  <h1>{site_name} — About the probabilistic flood emulator</h1>
  <a href="{main_page_url}" class="back-btn">← Back to flood map</a>
</header>

<main>

  <div class="section">
    <h2>Project overview</h2>
    <p>
      This dashboard is produced by the <strong>Uncertainty Quantification for Flood
      Modelling (UQ4FM)</strong> project at the University of Edinburgh. It demonstrates
      how surrogate modelling can make probabilistic flood mapping computationally
      tractable for operational use by practitioners and regulators.
    </p>
    <p>
      Traditional Full Monte Carlo (FMC) probabilistic analysis requires running a
      high-resolution 2D hydraulic model hundreds or thousands of times — prohibitively
      expensive for routine design work. The emulator approach replaces those expensive
      model runs with a fast surrogate trained on a small set of simulations.
    </p>
    <div class="placeholder">
      <p>[ Expand with project-specific context, funding acknowledgements, or partner logos here. ]</p>
    </div>
  </div>

  <div class="section">
    <h2>Surrogate modelling methodology</h2>
    <p>
      The surrogate used here is <strong>PCK (Polynomial Chaos Kriging)</strong>, which
      combines a Polynomial Chaos Expansion (PCE) systematic trend with a Gaussian
      Process Regression (GPR) residual. PCE provides an orthogonal polynomial
      approximation of the response surface; the GPR component captures residual
      non-linear patterns and provides a built-in variance estimate.
    </p>
    <p>
      To handle the high dimensionality of spatial flood maps (up to 900,000 grid cells),
      <strong>Proper Orthogonal Decomposition (POD)</strong> via Singular Value
      Decomposition (SVD) first compresses each map into a small number of dominant
      spatial modes. A separate PCK model is then fitted to each modal coefficient,
      reducing the learning problem from 900,000 outputs to ~10–20.
    </p>
    <ul>
      <li><strong>Training set:</strong> N=150 LISFLOOD-FP simulations, Sobol-sampled
          over the inflow uncertainty space.</li>
      <li><strong>Inputs:</strong> peak discharge at each tributary (Uniform, bounded
          by GEV 95% CI) + inter-tributary timing lag (Normal).</li>
      <li><strong>Output:</strong> maximum inundation depth map (1000 × 900 cells,
          5 m resolution).</li>
    </ul>
    <div class="placeholder">
      <p>[ Add validation metrics (R², MAE, spatial bias maps) and figures here. ]</p>
    </div>
  </div>

  <div class="section">
    <h2>Inflow uncertainty characterisation</h2>
    <p>
      Uncertainty in the peak inflow at each tributary is derived from a GEV analysis
      of the Annual Maximum (AMAX) flow series, with 95% bootstrap confidence intervals
      (1,000 resamples). The lower and upper CI bounds at the target return period
      define the Uniform distribution for each input.
    </p>
    <p>
      The timing lag between tributary peaks is characterised from historical 15-minute
      flow records, with a Normal distribution fitted to observed peak-to-peak offsets.
    </p>
    <div class="placeholder">
      <p>[ Insert GEV fit plots, lag distribution histograms, or a table of fitted
          parameters here. ]</p>
    </div>
  </div>

  <div class="section">
    <h2>How to read the probabilistic maps</h2>
    <ul>
      <li><strong>P(inundation):</strong> fraction of Monte Carlo runs in which a
          cell is inundated. 0.9 means the cell flooded in 90% of realisations.</li>
      <li><strong>Median depth (P50):</strong> the depth exceeded in 50% of runs —
          the best single-number summary.</li>
      <li><strong>P05 / P95 depths:</strong> near-best-case and near-worst-case
          estimates. P95 should be used for critical infrastructure and design.</li>
      <li><strong>P95−Median / P95−P05 range maps:</strong> hydrological sensitivity.
          Large values indicate that predictions are strongly uncertain — more freeboard
          may be warranted.</li>
      <li><strong>Depth exceedance curve (click any cell):</strong> shows P(depth &gt; d)
          vs d at that point. Read: "there is a X% chance depth exceeds Y m."</li>
    </ul>
  </div>

  <div class="section">
    <h2>References</h2>
    <p>
      Siripatana, A., Wilson, A. L., &amp; Beevers, L. (2025). Uncertainty quantification
      for multi-input fluvial flood inundation using GPR- and PCE-based surrogates.
      <em>Water Resources Research</em>, 61, e2024WR039668.
      <a href="https://doi.org/10.1029/2024WR039668"
         style="color:#60a5fa;">https://doi.org/10.1029/2024WR039668</a>
    </p>
    <div class="placeholder">
      <p>[ Add further references, data sources, or model documentation links here. ]</p>
    </div>
  </div>

  <div class="section">
    <h2>Contact &amp; acknowledgements</h2>
    <div class="placeholder">
      <p>[ Nina Fischer, Adil Siripatana, Amy Wilson, Lindsay Beevers —
          University of Edinburgh, School of Engineering. Add contact details,
          grant numbers, and acknowledgements here. ]</p>
    </div>
  </div>

</main>

<footer>
  UQ4FM — Uncertainty Quantification for Flood Modelling — University of Edinburgh
</footer>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved about page → {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB)")
    return out_path
