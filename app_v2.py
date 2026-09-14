from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.patches import Rectangle
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="SION Visual Field Progression - Stacked Bar",
    page_icon="👁️",
    layout="wide",
)

# Original SION settings requested for this prototype
BASELINE_VFS = 3
TOP_K = 3
REQUIRED_CONSECUTIVE = 2


# ============================================================
# 156-ARCHETYPE REGIONAL GROUPS
# ============================================================
def at_range(a: int, b: int) -> List[str]:
    return [f"AT{i}" for i in range(a, b + 1)]


AT_GROUPS = {
    "Normal": ["AT1"],
    "Superior": (
        at_range(39, 56)
        + at_range(81, 82)
        + at_range(91, 92)
        + ["AT95", "AT99", "AT102"]
        + at_range(105, 108)
        + at_range(115, 117)
        + at_range(119, 120)
        + at_range(123, 124)
        + at_range(129, 142)
    ),
    "Inferior": (
        at_range(57, 74)
        + at_range(83, 84)
        + at_range(93, 94)
        + ["AT98"]
        + at_range(100, 101)
        + at_range(109, 114)
        + ["AT118"]
        + at_range(121, 122)
        + at_range(125, 126)
        + at_range(143, 156)
    ),
    "Whole": (
        at_range(2, 38)
        + at_range(75, 80)
        + at_range(85, 90)
        + at_range(96, 97)
        + at_range(103, 104)
        + at_range(127, 128)
    ),
}


# ============================================================
# DATA VALIDATION / NORMALIZATION
# ============================================================
def normalize_eye(value) -> str:
    s = str(value).strip().lower()
    aliases = {
        "r": "Right", "right": "Right", "od": "Right",
        "l": "Left", "left": "Left", "os": "Left",
    }
    return aliases.get(s, str(value).strip())


def validate_input(df: pd.DataFrame) -> None:
    required = {"PatID", "Eye", "Age", "MD"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    missing_at = [f"AT{i}" for i in range(1, 157) if f"AT{i}" not in df.columns]
    if missing_at:
        raise ValueError(
            "The SION burden calculation expects AT1-AT156. "
            f"Missing {len(missing_at)} archetype columns; first few: {missing_at[:8]}"
        )


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    validate_input(df)
    df["Eye"] = df["Eye"].map(normalize_eye)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["MD"] = pd.to_numeric(df["MD"], errors="coerce")
    for i in range(1, 157):
        col = f"AT{i}"
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


# ============================================================
# REGIONAL BURDEN (TOP-K WITHIN EACH REGION)
# ============================================================
def topk_sum(row: pd.Series, cols: List[str], k: int = TOP_K) -> float:
    vals = pd.to_numeric(row[cols], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(vals) == 0:
        return 0.0
    k = min(k, len(vals))
    if k == len(vals):
        return float(vals.sum())
    idx = np.argpartition(vals, -k)[-k:]
    return float(vals[idx].sum())


def calculate_swin_burdens(df: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    out = df.copy()
    out["Normal_weight"] = pd.to_numeric(out["AT1"], errors="coerce").fillna(0.0)
    out["Superior_burden"] = out.apply(
        lambda r: topk_sum(r, AT_GROUPS["Superior"], top_k), axis=1
    )
    out["Inferior_burden"] = out.apply(
        lambda r: topk_sum(r, AT_GROUPS["Inferior"], top_k), axis=1
    )
    out["Whole_burden"] = out.apply(
        lambda r: topk_sum(r, AT_GROUPS["Whole"], top_k), axis=1
    )
    return out


# ============================================================
# ORIGINAL-SWIN-STYLE PERSISTENCE / EVENT LOGIC
# baseline = median of first 3 VFs
# required_consecutive = 2
# dynamic regional references after confirmed events
# ============================================================
def persists(mask: np.ndarray, start: int, required: int = REQUIRED_CONSECUTIVE) -> bool:
    end = start + required
    return end <= len(mask) and bool(np.all(mask[start:end]))


def detect_swin_progression(
    g: pd.DataFrame,
    superior_presence: float,
    inferior_presence: float,
    whole_presence: float,
    onset_delta: float,
    worsening_delta: float,
    baseline_vfs: int = BASELINE_VFS,
    required_consecutive: int = REQUIRED_CONSECUTIVE,
) -> Tuple[dict, pd.DataFrame]:
    """
    Detect persistent SWIN events for one patient-eye.

    Rules in this web prototype
    ---------------------------
    1) Baseline reference for S/I/W/N is the median of the first 3 VFs.
    2) A new S or I defect requires:
         - the current dynamic reference is below its presence threshold,
         - burden reaches the presence threshold,
         - burden increases by at least onset_delta from the reference,
         - the condition persists for required_consecutive VFs.
    3) Worsening in S/I/W requires:
         - the dynamic reference is already present,
         - burden rises by at least worsening_delta,
         - the condition persists for required_consecutive VFs.
    4) When a change is confirmed, the regional reference is updated to the
       median burden across the confirming VFs (dynamic reference).
    5) Normal-to-defect is reported when all baseline regional burdens are
       below their presence thresholds and the first persistent regional
       onset is confirmed.
    """
    g = g.sort_values("Age").reset_index(drop=True).copy()
    g["VF_number"] = np.arange(1, len(g) + 1)
    g["SWIN_event"] = ""
    g["SWIN_progression"] = False
    g["SWIN_confirmed"] = False

    if len(g) < baseline_vfs:
        return {
            "status": "Insufficient VFs",
            "progressing": False,
            "first_progression_time": None,
            "events": [],
            "baseline": {},
        }, g

    burden_map = {
        "Superior": "Superior_burden",
        "Inferior": "Inferior_burden",
        "Whole-field": "Whole_burden",
    }
    presence = {
        "Superior": superior_presence,
        "Inferior": inferior_presence,
        "Whole-field": whole_presence,
    }

    baseline = {
        "Normal": float(g.loc[: baseline_vfs - 1, "Normal_weight"].median()),
        "Superior": float(g.loc[: baseline_vfs - 1, "Superior_burden"].median()),
        "Inferior": float(g.loc[: baseline_vfs - 1, "Inferior_burden"].median()),
        "Whole-field": float(g.loc[: baseline_vfs - 1, "Whole_burden"].median()),
    }
    refs = {
        "Superior": baseline["Superior"],
        "Inferior": baseline["Inferior"],
        "Whole-field": baseline["Whole-field"],
    }

    baseline_all_clear = (
        baseline["Superior"] < superior_presence
        and baseline["Inferior"] < inferior_presence
        and baseline["Whole-field"] < whole_presence
    )

    events: List[dict] = []
    normal_to_defect_reported = False
    start_idx = baseline_vfs

    # Scan visits in chronological order. Once an event is confirmed for a
    # region, its dynamic reference is moved forward.
    i = start_idx
    while i < len(g):
        event_happened_at_i = False

        for region, col in burden_map.items():
            thr = presence[region]
            ref = refs[region]
            values = g[col].to_numpy(dtype=float)

            if ref < thr:
                mask = (values >= thr) & ((values - ref) >= onset_delta)
                event_name = (
                    "New superior defect" if region == "Superior"
                    else "New inferior defect" if region == "Inferior"
                    else "New whole-field defect"
                )
                event_type = "onset"
            else:
                mask = (values - ref) >= worsening_delta
                event_name = (
                    "Superior worsening" if region == "Superior"
                    else "Inferior worsening" if region == "Inferior"
                    else "Whole-field worsening"
                )
                event_type = "worsening"

            # Do not allow baseline VFs to trigger an event.
            mask[:start_idx] = False

            if persists(mask, i, required_consecutive):
                confirm_slice = slice(i, i + required_consecutive)
                confirm_vals = values[confirm_slice]
                confirmation_index = i + required_consecutive - 1
                event_age = float(g.loc[i, "Age"])
                confirmation_age = float(g.loc[confirmation_index, "Age"])

                events.append({
                    "event": event_name,
                    "region": region,
                    "type": event_type,
                    "first_age": event_age,
                    "confirmed_age": confirmation_age,
                    "vf_number": int(g.loc[i, "VF_number"]),
                    "confirmed_vf_number": int(g.loc[confirmation_index, "VF_number"]),
                })

                # Mark the onset VF and confirming VF(s)
                g.loc[i:confirmation_index, "SWIN_progression"] = True
                g.loc[confirmation_index, "SWIN_confirmed"] = True
                old = g.loc[confirmation_index, "SWIN_event"]
                g.loc[confirmation_index, "SWIN_event"] = (
                    event_name if not old else f"{old}; {event_name}"
                )

                # Optional overall Normal -> defect label at first regional onset
                if baseline_all_clear and not normal_to_defect_reported and event_type == "onset":
                    events.append({
                        "event": "Normal to defect",
                        "region": region,
                        "type": "onset",
                        "first_age": event_age,
                        "confirmed_age": confirmation_age,
                        "vf_number": int(g.loc[i, "VF_number"]),
                        "confirmed_vf_number": int(g.loc[confirmation_index, "VF_number"]),
                    })
                    normal_to_defect_reported = True

                # Dynamic regional reference after a confirmed change
                refs[region] = float(np.median(confirm_vals))
                event_happened_at_i = True

        i += 1

    # De-duplicate events that have identical name/region/time
    dedup = []
    seen = set()
    for e in sorted(events, key=lambda x: (x["first_age"], x["event"])):
        key = (e["event"], e["region"], e["first_age"], e["confirmed_age"])
        if key not in seen:
            dedup.append(e)
            seen.add(key)
    events = dedup

    if events:
        first_time = min(e["first_age"] for e in events)
        event_names = []
        for e in events:
            if e["event"] != "Normal to defect" and e["event"] not in event_names:
                event_names.append(e["event"])
        status = "Progressing"
    else:
        first_time = None
        event_names = []
        status = "No meaningful change"

    return {
        "status": status,
        "progressing": bool(events),
        "first_progression_time": first_time,
        "events": events,
        "event_names": event_names,
        "baseline": baseline,
        "final_references": refs,
    }, g


# ============================================================
# BEBIE CURVE
# ============================================================
def get_td_columns(df: pd.DataFrame, exclude=(26, 35)) -> List[str]:
    cols = []
    for i in range(1, 55):
        if i in exclude:
            continue
        for name in (f"TD_{i}", f"TD{i}"):
            if name in df.columns:
                cols.append(name)
                break
    return cols


def calculate_bebie_curve(row: pd.Series, td_cols: List[str]) -> np.ndarray:
    vals = pd.to_numeric(row[td_cols], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return np.sort(vals)[::-1]


def make_interactive_burden_figure(g: pd.DataFrame, baseline_vfs=BASELINE_VFS):
    """Interactive SWIN burden visualization as a stacked bar chart.

    Each visual-field visit is one bar positioned by age. The stacked segments
    show Normal, Superior, Inferior, and Whole-field burdens. Hovering over a
    segment shows Region, Age, Burden, and VF number. Legend items remain
    interactive: click to hide/show a region; double-click to isolate it.
    """
    fig = go.Figure()

    series = [
        ("Normal", "Normal_weight"),
        ("Superior", "Superior_burden"),
        ("Inferior", "Inferior_burden"),
        ("Whole-field", "Whole_burden"),
    ]

    ages = g["Age"].to_numpy(dtype=float)
    vf_numbers = g["VF_number"].to_numpy(dtype=int)

    for region, col in series:
        fig.add_trace(
            go.Bar(
                x=ages,
                y=g[col].to_numpy(dtype=float),
                name=region,
                customdata=np.column_stack([
                    np.repeat(region, len(g)),
                    vf_numbers,
                ]),
                hovertemplate=(
                    "Region: %{customdata[0]}<br>"
                    "Age: %{x:.1f}<br>"
                    "Burden: %{y:.4f}<br>"
                    "VF: %{customdata[1]:.0f}"
                    "<extra></extra>"
                ),
            )
        )

    if len(g) >= baseline_vfs:
        baseline_end = float(g.iloc[baseline_vfs - 1]["Age"])
        fig.add_vline(
            x=baseline_end,
            line_dash="dash",
            line_width=1.5,
            opacity=0.65,
            annotation_text="Baseline end",
            annotation_position="top",
        )

    # Give the stacked bars a little headroom above the largest visit total.
    stack_total = g[
        ["Normal_weight", "Superior_burden", "Inferior_burden", "Whole_burden"]
    ].sum(axis=1)
    ymax = max(1.0, float(stack_total.max()) * 1.10)

    fig.update_layout(
        title="Superior Inferior Other-field Normal (SION) — Stacked Bar",
        barmode="stack",
        xaxis_title="Age",
        yaxis_title="Archetypal weight / burden",
        yaxis=dict(range=[0, ymax]),
        hovermode="closest",
        bargap=0.2,
        legend=dict(
            title="Region",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=45, r=20, t=90, b=45),
        height=390,
    )
    return fig


def make_interactive_bebie_figure(g: pd.DataFrame):
    """Interactive serial Bebie curves using Plotly.

    By default, only the first and last VF curves are visible. Intermediate
    VFs remain listed in the legend and can be displayed by clicking their
    legend labels. Clicking a visible legend item hides it; clicking again
    shows it. Double-clicking isolates one VF curve.

    Hover shows Age, Rank, TD, and VF number.
    """
    td_cols = get_td_columns(g)
    if not td_cols:
        raise ValueError("No TD columns found. Expected TD_1...TD_54 (or TD1...TD54).")

    fig = go.Figure()
    n_vfs = len(g)

    for row_position, (_, row) in enumerate(g.iterrows()):
        bebie = calculate_bebie_curve(row, td_cols)
        rank = np.arange(1, len(bebie) + 1)
        age = float(row["Age"])
        vf_number = int(row["VF_number"])

        # Show only the first and last VF initially.
        # Plotly's "legendonly" keeps the other curves in the legend so the
        # user can turn them on interactively.
        is_first = row_position == 0
        is_last = row_position == n_vfs - 1
        initial_visibility = True if (is_first or is_last) else "legendonly"

        fig.add_trace(
            go.Scatter(
                x=rank,
                y=bebie,
                mode="lines+markers",
                name=f"VF{vf_number} (Age {age:.1f})",
                visible=initial_visibility,
                customdata=np.column_stack([
                    np.repeat(age, len(rank)),
                    np.repeat(vf_number, len(rank)),
                ]),
                hovertemplate=(
                    "Age: %{customdata[0]:.1f}<br>"
                    "Rank: %{x:.0f}<br>"
                    "TD: %{y:.2f} dB<br>"
                    "VF: %{customdata[1]:.0f}"
                    "<extra></extra>"
                ),
                line=dict(width=1.8),
                marker=dict(size=5),
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_width=1, opacity=0.5)

    fig.update_layout(
        title="Serial Bebie Curves Based on Total Deviation",
        xaxis_title="Ranked visual-field location (least damaged → most damaged)",
        yaxis_title="Total deviation (dB)",
        hovermode="closest",
        legend=dict(
            title="Visual fields — click to show/hide",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=45, r=20, t=70, b=55),
        height=470,
    )
    return fig


def make_interactive_md_slope_figure(g: pd.DataFrame):
    """Interactive longitudinal MD plot with a linear MD slope.

    The observed MD values are connected in chronological order. A least-
    squares regression line is fitted against Age when at least two valid
    observations with different ages are available. Hovering over observed
    points shows Age, MD, and VF number.
    """
    plot_df = g[["Age", "MD", "VF_number"]].copy()
    plot_df["Age"] = pd.to_numeric(plot_df["Age"], errors="coerce")
    plot_df["MD"] = pd.to_numeric(plot_df["MD"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Age", "MD"]).sort_values("Age")

    if plot_df.empty:
        raise ValueError("No valid Age/MD values found for the MD slope plot.")

    ages = plot_df["Age"].to_numpy(dtype=float)
    md = plot_df["MD"].to_numpy(dtype=float)
    vf_numbers = plot_df["VF_number"].to_numpy(dtype=int)

    fig = go.Figure()

    # Observed longitudinal MD values.
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=md,
            mode="lines+markers",
            name="Observed MD",
            customdata=vf_numbers.reshape(-1, 1),
            hovertemplate=(
                "Age: %{x:.1f}<br>"
                "MD: %{y:.2f} dB<br>"
                "VF: %{customdata[0]:.0f}"
                "<extra></extra>"
            ),
            line=dict(width=2),
            marker=dict(size=7),
        )
    )

    slope = None
    intercept = None

    # Fit MD = slope * Age + intercept. Require different age values so the
    # regression is well-defined.
    if len(plot_df) >= 2 and np.ptp(ages) > 0:
        slope, intercept = np.polyfit(ages, md, 1)
        fit_x = np.array([ages.min(), ages.max()], dtype=float)
        fit_y = slope * fit_x + intercept

        fig.add_trace(
            go.Scatter(
                x=fit_x,
                y=fit_y,
                mode="lines",
                name=f"Linear slope ({slope:.2f} dB/year)",
                hovertemplate=(
                    "Regression line<br>"
                    f"Slope: {slope:.3f} dB/year"
                    "<extra></extra>"
                ),
                line=dict(width=2, dash="dash"),
            )
        )

    title = "Mean Deviation (MD) Over Time"
    if slope is not None:
        title += f" — Slope: {slope:.2f} dB/year"

    fig.update_layout(
        title=title,
        xaxis_title="Age",
        yaxis_title="MD (dB)",
        hovermode="closest",
        legend=dict(
            title="MD",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=45, r=20, t=70, b=55),
        height=360,
    )

    return fig


# ============================================================
# 24-2 SENSITIVITY MAP
# ============================================================
def get_sensitivity_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for i in range(1, 55):
        if i in (26, 35):
            continue
        for name in (f"Sens_{i}", f"Sens{i}"):
            if name in df.columns:
                cols.append(name)
                break
    return cols


def sensitivity_to_9x9(row: pd.Series, sens_cols: List[str], fill_value=45.0) -> np.ndarray:
    """Reproduce the user's 24-2 -> 9x9 padding scheme in memory."""
    vals = pd.to_numeric(row[sens_cols], errors="coerce").fillna(fill_value).tolist()

    # Same insertion positions as add_zeros_for24_2().
    inserts = [
        (0, fill_value), (1, fill_value), (2, fill_value),
        (7, fill_value), (8, fill_value), (9, fill_value), (10, fill_value),
        (17, fill_value), (18, fill_value),
        (34, 0.0), (43, 0.0),
        (45, fill_value), (54, fill_value), (55, fill_value),
        (62, fill_value), (63, fill_value), (64, fill_value), (65, fill_value),
        (70, fill_value), (71, fill_value), (72, fill_value), (73, fill_value),
        (74, fill_value), (75, fill_value), (76, fill_value), (77, fill_value),
        (78, fill_value), (79, fill_value), (80, fill_value),
    ]
    for idx, value in inserts:
        vals.insert(idx, value)

    if len(vals) != 81:
        raise ValueError(f"Expected 81 values after padding, got {len(vals)}")
    return np.asarray(vals, dtype=float).reshape(9, 9)


def plot_result_onebyone_imshow_fast(
    fname,
    dat,
    *,
    td=9,
    vmin=0,
    vcenter=30,
    vmax=60,
    fwidth=1.5,
    fheight=1.0,
    annot=False,
    bsgrey=False,
    blind_coords=((3, 7), (4, 7)),
    pad_top=0.10,
    pad_left=0.10,
    pad_right=0.10,
    pad_bottom=0.00,
    dpi=150,
    grid=True,
    grid_linewidth=0.5,
    grid_alpha=0.08,
    grid_color="black",
):
    """
    User-provided fast visual-field plotting function.

    Saves one PNG per visual field as:
        <fname>_000001.png, <fname>_000002.png, ...
    """
    fname = str(fname)
    Path(fname).parent.mkdir(parents=True, exist_ok=True)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "CustomRedWhiteBlue", ["#ff0000", "#ffffff", "#a9def9"]
    )
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    dat = np.asarray(dat)
    if dat.ndim == 2:
        if dat.shape[1] != td * td:
            raise ValueError(
                f"Expected dat.shape[1] == td*td ({td*td}), got {dat.shape[1]}"
            )
        dat_3d = dat.reshape(dat.shape[0], td, td)
    elif dat.ndim == 3:
        dat_3d = dat
    else:
        raise ValueError("dat must be 2D or 3D")

    fig, ax = plt.subplots(
        figsize=(fwidth, fheight), dpi=dpi, constrained_layout=False
    )

    ax.set_position([
        pad_left,
        pad_bottom,
        1.0 - pad_left - pad_right,
        1.0 - pad_top - pad_bottom,
    ])
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    extent = (-0.5, td - 0.5, td - 0.5, -0.5)

    im = ax.imshow(
        dat_3d[0],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
        extent=extent,
    )

    if grid:
        ax.set_xticks(np.arange(-0.5, td, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, td, 1), minor=True)
        ax.grid(
            which="minor",
            linewidth=grid_linewidth,
            color=grid_color,
            alpha=grid_alpha,
        )

    ax.set_xlim(-0.5, td - 0.5)
    ax.set_ylim(td - 0.5, -0.5)

    texts = []
    if annot:
        for r in range(td):
            for c in range(td):
                texts.append(
                    ax.text(c, r, "", ha="center", va="center", fontsize=6)
                )

    if bsgrey:
        overlay = np.full((td, td), np.nan, dtype=float)
        for r, c in blind_coords:
            overlay[r, c] = 1.0
        ax.imshow(
            overlay,
            cmap=ListedColormap(["grey"]),
            interpolation="nearest",
            origin="upper",
            extent=extent,
            vmin=0,
            vmax=1,
        )

    for x in range(dat_3d.shape[0]):
        arr = dat_3d[x]
        im.set_data(arr)

        if annot:
            k = 0
            for r in range(td):
                for c in range(td):
                    texts[k].set_text(f"{arr[r, c]:.2f}")
                    k += 1

        fig.savefig(f"{fname}_{x+1:06d}.png", dpi=dpi, bbox_inches=None)

    plt.close(fig)


def draw_sensitivity_map_on_axis(
    ax,
    arr,
    *,
    td=9,
    vmin=0,
    vcenter=30,
    vmax=60,
    bsgrey=True,
    blind_coords=((3, 7), (4, 7)),
    grid=True,
    grid_linewidth=1,
    grid_alpha=0.08,
    grid_color="black",
    border=True,
    border_color="#222222",
    border_linewidth=1.4,
):
    """Draw the same visual style directly on a Streamlit Matplotlib axis.

    This avoids writing temporary PNGs for every rerun while preserving the
    colormap, normalization, grid, and blind-spot overlay of
    plot_result_onebyone_imshow_fast().

    A thin outer border is added around each 9x9 sensitivity map so that each
    VF image looks clearly separated in the web layout.
    """
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "CustomRedWhiteBlue", ["#ff0000", "#ffffff", "#a9def9"]
    )
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    extent = (-0.5, td - 0.5, td - 0.5, -0.5)

    ax.set_facecolor("white")
    ax.imshow(
        arr,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
        extent=extent,
    )

    if grid:
        ax.set_xticks(np.arange(-0.5, td, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, td, 1), minor=True)
        ax.grid(
            which="minor",
            linewidth=grid_linewidth,
            color=grid_color,
            alpha=grid_alpha,
        )

    if bsgrey:
        overlay = np.full((td, td), np.nan, dtype=float)
        for r, c in blind_coords:
            overlay[r, c] = 1.0
        ax.imshow(
            overlay,
            cmap=ListedColormap(["grey"]),
            interpolation="nearest",
            origin="upper",
            extent=extent,
            vmin=0,
            vmax=1,
        )

    if border:
        ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                td,
                td,
                fill=False,
                edgecolor=border_color,
                linewidth=border_linewidth,
                zorder=10,
                joinstyle="miter",
            )
        )

    ax.set_xlim(-0.5, td - 0.5)
    ax.set_ylim(td - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    for spine in ax.spines.values():
        spine.set_visible(False)


def make_vf_strip_figure(g: pd.DataFrame):
    """Create the longitudinal sensitivity-map strip for the web app.

    Uses the user's exact red-white-blue plotting convention:
      red = lower sensitivity, white = around 30 dB, blue = higher sensitivity,
    with grey blind spots at the supplied 9x9 coordinates.

    Each VF is drawn with a visible outer border so the image panel is clearly
    separated from surrounding text and neighboring VFs.
    """
    sens_cols = get_sensitivity_columns(g)
    if len(sens_cols) != 52:
        raise ValueError(
            f"Expected 52 sensitivity columns (excluding blind spots), found {len(sens_cols)}."
        )

    n = len(g)
    fig, axes = plt.subplots(
        n, 1,
        figsize=(4.1, max(1.35, n * 1.02)),
        constrained_layout=False,
    )
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, g.iterrows()):
        mat = sensitivity_to_9x9(row, sens_cols)
        draw_sensitivity_map_on_axis(
            ax,
            mat,
            td=9,
            vmin=0,
            vcenter=30,
            vmax=60,
            bsgrey=True,
            grid=True,
            grid_linewidth=1,
            grid_alpha=0.08,
            border=True,
            border_linewidth=1.4,
        )
        ax.set_title(
            f"VF{int(row['VF_number'])}   Age {row['Age']:.1f}   MD {row['MD']:.2f}",
            fontsize=8.5,
            loc="left",
            pad=4,
        )

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.06, right=0.98, top=0.985, bottom=0.015, hspace=0.42
    )
    return fig


def make_single_sensitivity_figure(row: pd.Series, sens_cols: List[str]):
    """Create one compact sensitivity-map figure with an outside border."""
    mat = sensitivity_to_9x9(row, sens_cols)

    fig, ax = plt.subplots(figsize=(1.35, 1.35), dpi=150)
    draw_sensitivity_map_on_axis(
        ax,
        mat,
        td=9,
        vmin=0,
        vcenter=30,
        vmax=60,
        bsgrey=True,
        grid=True,
        grid_linewidth=1,
        grid_alpha=0.08,
        border=True,
        border_color="black",
        border_linewidth=1.5,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    return fig


def render_vf_age_md_image_table(g: pd.DataFrame):
    """Render Age, MD, and sensitivity image side-by-side for each VF."""
    sens_cols = get_sensitivity_columns(g)
    if len(sens_cols) != 52:
        raise ValueError(
            f"Expected 52 sensitivity columns (excluding blind spots), found {len(sens_cols)}."
        )

    st.markdown("#### Longitudinal visual fields")

    # Header row
    h_age, h_md, h_img = st.columns([0.8, 0.8, 1.4], gap="small")
    h_age.markdown("**Age**")
    h_md.markdown("**MD**")
    h_img.markdown("**Image**")
    st.markdown("<hr style='margin:0.1rem 0 0.35rem 0;'>", unsafe_allow_html=True)

    for _, row in g.iterrows():
        c_age, c_md, c_img = st.columns([0.8, 0.8, 1.4], gap="small")

        # Use a fixed-height flex box so Age and MD align vertically with the map.
        c_age.markdown(
            f"<div style='height:118px; display:flex; align-items:center; "
            f"justify-content:center; font-size:0.92rem;'>{row['Age']:.1f}</div>",
            unsafe_allow_html=True,
        )
        c_md.markdown(
            f"<div style='height:118px; display:flex; align-items:center; "
            f"justify-content:center; font-size:0.92rem;'>{row['MD']:.2f}</div>",
            unsafe_allow_html=True,
        )

        fig = make_single_sensitivity_figure(row, sens_cols)
        c_img.pyplot(fig, use_container_width=False)
        plt.close(fig)

        st.markdown("<hr style='margin:0.15rem 0 0.3rem 0; opacity:0.25;'>", unsafe_allow_html=True)


# ============================================================
# STREAMLIT UI
# ============================================================
st.title("SION Visual Field Progression — Stacked Bar")
st.caption(
    "Load a longitudinal CSV, select a patient-eye, set SION thresholds, "
    "and generate SION burden, Bebie curves, visual-field maps, and progression events."
)

uploaded = st.file_uploader("Input file", type=["csv"], help="Expected format: PatID, Eye, Age, MD, AT1-AT156, sensitivity and TD columns.")

if uploaded is None:
    st.info("Upload the input CSV to begin.")
    st.stop()

try:
    df = prepare_input(pd.read_csv(uploaded))
    df = calculate_swin_burdens(df, top_k=TOP_K)
except Exception as exc:
    st.error(f"Could not read the input file: {exc}")
    st.stop()

left, middle, right = st.columns([0.23, 0.27, 0.50], gap="large")

with left:
    st.subheader("Patient selection")
    patient_ids = sorted(df["PatID"].dropna().unique().tolist(), key=lambda x: str(x))
    patient_id = st.selectbox("Patient ID", patient_ids)

    eye_options = sorted(df.loc[df["PatID"] == patient_id, "Eye"].dropna().unique().tolist())
    eye = st.selectbox("Eye", eye_options)

    st.subheader("Thresholds")
    superior_presence = st.number_input(
        "Superior presence", min_value=0.02, max_value=0.12, value=0.04, step=0.01, format="%.2f"
    )
    inferior_presence = st.number_input(
        "Inferior presence", min_value=0.02, max_value=0.12, value=0.04, step=0.01, format="%.2f"
    )
    whole_presence = st.number_input(
        "Whole-field presence", min_value=0.02, max_value=0.12, value=0.12, step=0.01, format="%.2f"
    )
    onset_delta = st.number_input(
        "Onset delta", min_value=0.02, max_value=0.12, value=0.03, step=0.01, format="%.2f"
    )
    worsening_delta = st.number_input(
        "Worsening delta", min_value=0.02, max_value=0.12, value=0.10, step=0.01, format="%.2f"
    )

    st.caption(
        f"Fixed SION settings: baseline = first {BASELINE_VFS} VFs (median), "
        f"top-K = {TOP_K}, persistence = {REQUIRED_CONSECUTIVE} consecutive VFs."
    )

    run = st.button("Run", type="primary", use_container_width=True)

# Keep the selected patient-eye available even before Run for a quick data preview.
selected = (
    df[(df["PatID"] == patient_id) & (df["Eye"] == eye)]
    .sort_values("Age")
    .reset_index(drop=True)
)
selected["VF_number"] = np.arange(1, len(selected) + 1)

with middle:
    try:
        render_vf_age_md_image_table(selected)
    except Exception as exc:
        st.warning(f"Sensitivity maps could not be drawn: {exc}")

if run:
    try:
        result, g = detect_swin_progression(
            selected,
            superior_presence=superior_presence,
            inferior_presence=inferior_presence,
            whole_presence=whole_presence,
            onset_delta=onset_delta,
            worsening_delta=worsening_delta,
        )
    except Exception as exc:
        st.error(f"SION processing failed: {exc}")
        st.stop()

    with right:
        st.subheader("Result")
        if result["progressing"]:
            st.error("Progressing")
            st.write(f"**First progression time:** {result['first_progression_time']:.1f}")
            if result["event_names"]:
                st.write("**All events:** " + ", ".join(result["event_names"]))
        else:
            st.success(result["status"])
            st.write("**First progression time:** —")
            st.write("**All events:** —")

        if result["events"]:
            with st.expander("Event details"):
                event_df = pd.DataFrame(result["events"])
                st.dataframe(event_df, hide_index=True, use_container_width=True)

    with right:
        try:
            burden_fig = make_interactive_burden_figure(g)
            st.plotly_chart(
                burden_fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True,
                },
            )

            bebie_fig = make_interactive_bebie_figure(g)
            st.plotly_chart(
                bebie_fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True,
                },
            )

            md_fig = make_interactive_md_slope_figure(g)
            st.plotly_chart(
                md_fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True,
                },
            )
        except Exception as exc:
            st.error(f"Visualization failed: {exc}")

        # Baseline reference information if needed
        with st.expander("Baseline / dynamic references"):
            st.write("Baseline median (first 3 VFs)")
            st.json({k: round(v, 4) for k, v in result["baseline"].items()})
            st.write("Final dynamic regional references")
            st.json({k: round(v, 4) for k, v in result.get("final_references", {}).items()})
        
        # Useful per-VF output table beneath the plots
        with st.expander("Per-VF SION values"):
            show_cols = [
                "VF_number", "Age", "MD",
                "Normal_weight", "Superior_burden", "Inferior_burden", "Whole_burden",
                "SWIN_event", "SWIN_progression", "SWIN_confirmed",
            ]
            st.dataframe(g[show_cols], hide_index=True, use_container_width=True)
else:
    with right:
        st.info("Choose thresholds and click **Run** to generate the SION result and visualizations.")
