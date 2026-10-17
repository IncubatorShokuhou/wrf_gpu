"""Public README variant of the WN3 deviation-band figure (light + dark), from the same sealed wn3_bands.csv.

CPU only. Run: MPLCONFIGDIR=/your/scratch python render_readme.py
Same data as render.py's wn3_deviation_bands; only layout, titles and plain-language labels differ.
No model execution, scientific-file write, tolerance change or archive read.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
FIELDS = [('T2', '2 m temperature'), ('PSFC', 'Surface pressure'), ('U10', '10 m wind (west–east)'),
          ('V10', '10 m wind (south–north)'), ('RAINNC', 'Accumulated rain'), ('T', 'Potential temperature'),
          ('U', 'Wind, west–east'), ('V', 'Wind, south–north'), ('W', 'Vertical wind'), ('QVAPOR', 'Water vapour')]
DOMAINS = [('d01', '9 km'), ('d02', '3 km'), ('d03', '1 km')]


def main():
    with (OUT / 'wn3_bands.csv').open() as f:
        rows = list(csv.DictReader(f))
    for dark in (False, True):
        bg, fg, muted = ('#0d1117', '#e6edf3', '#9aa7b4') if dark else ('#ffffff', '#1f2933', '#5b6b7b')
        colors = ['#58a6ff', '#f0883e', '#3fb950'] if dark else ['#0072b2', '#d55e00', '#009e73']
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': fg, 'axes.labelcolor': fg,
                             'xtick.color': muted, 'ytick.color': muted, 'axes.edgecolor': muted}):
            fig, axes = plt.subplots(2, 5, figsize=(16, 6.6), dpi=130)
            fig.set_facecolor(bg)
            for ax, (field, label) in zip(axes.flat, FIELDS):
                ax.set_facecolor(bg)
                limit = 0.0
                for (dom, name), color in zip(DOMAINS, colors):
                    v = [x for x in rows if x['field'] == field and x['domain'] == dom]
                    t = [float(x['lead_h']) for x in v]
                    ax.fill_between(t, 0, [float(x['p95_abs']) for x in v], color=color, alpha=.13, lw=0)
                    ax.plot(t, [float(x['rmse']) for x in v], color=color, lw=1.8, label=name)
                    limit = max(limit, max(float(x['rmse_limit']) for x in v))
                    unit = v[0]['unit']
                ax.axhline(limit, color=fg, ls=':', lw=1.2)
                ax.text(23.6, limit * .97, 'limit', ha='right', va='top', fontsize=9, color=fg)
                ax.set_ylim(0, limit * 1.08)
                ax.set_xlim(0, 24)
                ax.set_xticks([0, 6, 12, 18, 24])
                ax.set_title(f'{label} ({unit})', fontsize=10.5, color=fg)
                ax.grid(alpha=.15, color=muted)
                for s in ('top', 'right'):
                    ax.spines[s].set_visible(False)
            for ax in axes[1]:
                ax.set_xlabel('Forecast hour')
            axes[0, 0].legend(title='Domain', fontsize=9, title_fontsize=9, frameon=False, loc='center left')
            fig.suptitle('wrf_gpu versus original CPU-WRF: 24-hour Tenerife forecast, three nested domains',
                         x=.01, ha='left', fontsize=15, weight='bold', color=fg)
            fig.text(.01, .905, 'Lines: hourly difference over the whole domain (RMSE). Shading: 95 % of grid cells '
                     'differ by less. Dotted: fixed acceptance limit, set before the comparison.', fontsize=10.5,
                     color=muted)
            fig.text(.01, .012, 'Same inputs for both models, start 27 Feb 2026 18 UTC, v0.4.0 default physics. '
                     'Shading is a spatial spread, not a confidence interval.', fontsize=9, color=muted)
            fig.tight_layout(rect=(0, .035, 1, .89))
            p = OUT / ('readme_validation' + ('_dark' if dark else '') + '.png')
            fig.savefig(p, facecolor=bg)
            plt.close(fig)
            if p.stat().st_size >= 1_000_000:
                raise ValueError(f'{p.name} exceeds 1 MB')


if __name__ == '__main__':
    main()
