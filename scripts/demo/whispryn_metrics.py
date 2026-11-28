"""Run the existing Whispryn dashboard with offline and live RF evaluation."""
import argparse
import csv
import json
from pathlib import Path
from threading import Lock
from time import monotonic as clock

LABELS = ('background', 'clap', 'hey', 'whistle')
CSV = Path(__file__).resolve().parents[2] / 'runtime/training/rf_predictions.csv'


# 1. Read the offline benchmark. Rows = actual, columns = predicted.
def empty():
    return [[0] * len(LABELS) for _ in LABELS]


def read_csv(path):
    m = empty()
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = csv.DictReader(f)
        if not {'actual', 'predicted'} <= set(rows.fieldnames or []):
            raise ValueError('CSV must contain actual and predicted columns')

        for row in rows:
            a, p = ((row.get(k) or '').strip().lower()
                    for k in ('actual', 'predicted'))
            if a not in LABELS or p not in LABELS:
                raise ValueError(
                    f'Unknown label at CSV line {rows.line_num}: {a!r}, {p!r}'
                )
            m[LABELS.index(a)][LABELS.index(p)] += 1

    if not sum(map(sum, m)):
        raise ValueError('CSV has no predictions')
    return m


# 2. Calculate metrics. A dash means there is no denominator yet.
def pct(a, b):
    return f'{a / b:.1%}' if b else '—'


def summary(m):
    n = sum(map(sum, m))
    right = sum(m[i][i] for i in range(len(m)))
    return f'{right}/{n} correct | Accuracy: {pct(right, n)} | Errors: {n - right}'


def table(m):
    rows = [
        '<tr><th>Class</th><th>Trials</th><th>Precision</th>'
        '<th>Recall</th><th>F1</th></tr>'
    ]
    for i, name in enumerate(LABELS):
        tp = m[i][i]                          # Correct predictions for this class
        actual = sum(m[i])                    # Times this sound actually occurred
        predicted = sum(row[i] for row in m)  # Times the model named this sound

        values = (
            name, actual,
            pct(tp, predicted),               # Precision
            pct(tp, actual),                  # Recall
            pct(2 * tp, actual + predicted),  # F1
        )
        rows.append('<tr>' + ''.join(f'<td>{v}</td>' for v in values) + '</tr>')

    header = (
        '<tr><th>Actual / predicted</th>'
        + ''.join(f'<th>{x}</th>' for x in LABELS)
        + '</tr>'
    )
    grid = ''.join(
        '<tr><th>' + label + '</th>'
        + ''.join(f'<td>{n}</td>' for n in row) + '</tr>'
        for label, row in zip(LABELS, m)
    )
    return (
        f'<p><b>{summary(m)}</b></p><table>{"".join(rows)}</table>'
        f'<details><summary>Confusion matrix</summary>'
        f'<table>{header}{grid}</table></details>'
    )


# 3. The panel added to the existing dashboard.
PANEL = '''
<section class="card" id="rf-eval">
<style>
#rf-eval table {border-collapse:collapse; margin:8px 0}
#rf-eval td, #rf-eval th {padding:5px 9px; text-align:left; border-bottom:1px solid #ddd}
#rf-eval button {margin:3px; padding:6px 10px}
</style>
<h2>How often is Whispryn right?</h2>
<h3>Offline benchmark · Random Forest v2</h3>
OFFLINE_TABLE
<h3>This live session</h3>
<p>Choose the real sound first. Wait for the cue, then make it once.</p>
<div id="rf-buttons">TRIAL_BUTTONS</div>
<p id="rf-msg" role="status">Connecting…</p>
<div id="rf-results"></div>
<small id="rf-note"></small>
</section>
<script>
(() => {
  const find = id => document.getElementById(id);
  const buttons = [...find('rf-buttons').querySelectorAll('button')];
  let previous = '', posting = false;

  async function refresh() {
    try {
      const r = await fetch('/metrics', {cache:'no-store'});
      if (!r.ok) throw Error('Metrics unavailable');
      const s = await r.json();

      find('rf-msg').textContent = s.message;
      if (s.html !== previous) {
        find('rf-results').innerHTML = s.html;
        previous = s.html;
      }
      find('rf-note').textContent =
        'Unscored trials: ' + s.unscored + ' · A dash means not defined.';
      buttons.forEach(b => b.disabled = posting || !s.ready);
    } catch (e) {
      find('rf-msg').textContent = e.message;
      buttons.forEach(b => b.disabled = true);
    } finally {
      setTimeout(refresh, 300);
    }
  }

  buttons.forEach(b => b.onclick = async () => {
    posting = true;
    buttons.forEach(x => x.disabled = true);
    try {
      const r = await fetch('/metrics/' + b.value, {
        method:'POST', headers:{'X-Trial':'1'}
      });
      if (!r.ok) throw Error('Trial not started; check the dashboard status.');
    } catch (e) {
      find('rf-msg').textContent = e.message;
    } finally {
      posting = false;
    }
  });

  refresh();
})();
</script>'''


# 4. Count one fresh prediction per labeled trial, not one per browser refresh.
def install(demo, offline):
    if (demo.RATE, demo.CHUNK_SAMPLES, demo.RF_SAMPLES) != (16000, 8000, 48000):
        raise ValueError('Unexpected audio window size; check live_demo.py before scoring.')
    if tuple(demo.RF_CLASSES) != LABELS or '</main>' not in demo.HOME:
        raise ValueError('Unexpected dashboard layout or RF labels.')

    counts = empty()
    lock = Lock()
    target = last = cue = None
    steps = unscored = 0
    message = 'Waiting for the classifier.'
    original = demo.set_state

    def cancel(reason):
        nonlocal target, last, unscored, message
        if target is not None:
            unscored += 1
        target = last = None
        message = reason

    def expire(now):
        if target is not None and last is not None and now - last > 3:
            cancel('Audio stalled. Trial not scored.')

    def on_prediction(**state):
        nonlocal target, last, cue, steps, message
        original(**state)  # Preserve the existing dashboard updates.

        with lock:
            now = clock()
            expire(now)

            if state.get('error') or state.get('status') == 'Error':
                cancel('Classifier error. Check the microphone status.')
                return

            label = state.get('rf_name')
            if state.get('status') != 'Listening' or label not in LABELS:
                return

            if last is None and target is None:
                message = 'Ready. Choose the sound for your next trial.'
            last = now

            if target is None:
                return

            if cue is None:
                # This first prediction still includes audio from before the trial.
                cue = now
                message = ('Stay quiet now.' if target == 'background'
                           else f'Make the {target} sound now.')
                return

            steps += 1
            if steps == 6:
                # Six new half-second chunks replace the full three-second window.
                if not 2.55 <= now - cue <= 5:
                    cancel('Irregular audio timing. Trial not scored.')
                    return

                counts[LABELS.index(target)][LABELS.index(label)] += 1
                verdict = 'correct' if target == label else 'incorrect'
                message = f'Actual: {target} → Predicted: {label} ({verdict}).'
                target = None

    class Handler(demo.Dashboard):
        def do_GET(self):
            if self.path != '/metrics':
                return super().do_GET()  # Keep the original dashboard and camera routes.

            with lock:
                now = clock()
                expire(now)
                ready = target is None and last is not None and now - last <= 3
                data = dict(message=message, ready=ready,
                            unscored=unscored, html=table(counts))

            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            nonlocal target, cue, steps, message
            if self.headers.get('X-Trial') != '1':
                return self.send_error(403)
            if self.path not in ['/metrics/' + name for name in LABELS]:
                return self.send_error(404)

            with lock:
                now = clock()
                expire(now)
                if target is not None or last is None or now - last > 3:
                    return self.send_error(409, 'Trial busy or classifier not ready')
                target = self.path.split('/')[-1]
                cue, steps = None, 0
                message = 'Get ready. Wait for the cue.'

            self.send_response(202)
            self.send_header('Content-Length', '0')
            self.end_headers()

    buttons = ''.join(
        f'<button type="button" disabled value="{x}">Test {x}</button>'
        for x in LABELS
    )
    # Load the three views and insert the real benchmark and trial buttons.
    layout_path = Path(__file__).with_name("dashboard_pages.html")
    layout = layout_path.read_text(encoding="utf-8")
    layout = layout.replace("OFFLINE_TABLE", table(offline))
    layout = layout.replace("TRIAL_BUTTONS", buttons)

    # Keep the original header, Listening indicator, and live-update JavaScript.
    header, opening, rest = demo.HOME.partition("<main>")
    _, closing, tail = rest.partition("</main>")
    if not opening or not closing:
        raise ValueError("Could not find <main> in live_demo.py")

    # Reuse the tested trial-button and metrics-refresh code, not the old panel.
    _, script_tag, trial_script = PANEL.partition("<script>")
    if not script_tag:
        raise ValueError("Could not find the existing live-trial script")
    demo.HOME = header + opening + layout + closing + script_tag + trial_script + tail
    demo.Dashboard, demo.set_state = Handler, on_prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--csv', type=Path, default=CSV)
    args = parser.parse_args()

    try:
        offline = read_csv(args.csv)
    except (OSError, ValueError, csv.Error) as e:
        parser.exit(1, f'Benchmark unavailable: {e}\n')

    print(summary(offline))
    if not args.check:
        import live_demo
        install(live_demo, offline)
        live_demo.main()
