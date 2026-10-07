/* ==========================================================================
   Student Wellbeing ML Demo — frontend
   Talks to POST /predict. Never displays text as if it were a diagnosis.
   ========================================================================== */
'use strict';

const form = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const resetBtn = document.getElementById('resetBtn');
const retryBtn = document.getElementById('retryBtn');

const resultSection = document.getElementById('resultSection');
const errorPanel = document.getElementById('errorPanel');
const formStatus = document.getElementById('formStatus');

const feedbackEl = document.getElementById('feedback');
const charCount = document.getElementById('charCount');

/* Numeric fields: id -> { outputId, unit, decimals } */
const SLIDERS = {
    marks: { outputId: 'marks-value', unit: '%', decimals: 0 },
    attendance: { outputId: 'attendance-value', unit: '%', decimals: 0 },
    assignment_delay: { outputId: 'assignment_delay-value', unit: '', decimals: 0 },
    sleep_hours: { outputId: 'sleep_hours-value', unit: ' h', decimals: 1 },
    screen_time: { outputId: 'screen_time-value', unit: ' h', decimals: 1 },
};

/* The API's canonical level names -> how we label them in the UI.
   Deliberately "pattern" rather than "risk", to avoid sounding clinical. */
const LEVEL_LABELS = {
    LOW: 'Low pattern',
    MEDIUM: 'Medium pattern',
    HIGH: 'Higher-risk pattern',
};

/* Class order used by the branch breakdown grid: [API name, UI label]. */
const BRANCH_ROWS = [
    ['LOW', 'Low pattern'],
    ['MEDIUM', 'Medium pattern'],
    ['HIGH', 'Higher-risk pattern'],
];

/* ---------------------------------------------------------------------
   Slider live values
   --------------------------------------------------------------------- */
Object.entries(SLIDERS).forEach(([id, cfg]) => {
    const input = document.getElementById(id);
    const out = document.getElementById(cfg.outputId);
    if (!input || !out) return;

    const render = () => {
        const value = parseFloat(input.value).toFixed(cfg.decimals);
        out.textContent = value + cfg.unit;
    };
    input.addEventListener('input', render);
    render();
});

/* Character counter for the reflection textarea */
if (feedbackEl && charCount) {
    const updateCount = () => {
        charCount.textContent = String(feedbackEl.value.length);
    };
    feedbackEl.addEventListener('input', updateCount);
    updateCount();
}

/* ---------------------------------------------------------------------
   UI state helpers — every state is explicit, so there is never a
   blank screen.
   --------------------------------------------------------------------- */
function setStatus(kind, messages) {
    if (!kind) { formStatus.innerHTML = ''; return; }

    const box = document.createElement('div');
    box.className = 'status-message status-' + kind;

    const first = document.createElement('p');
    first.style.margin = '0';
    first.textContent = kind === 'invalid'
        ? 'Please fix the following before running the demo:'
        : 'Ready to run.';
    box.appendChild(first);

    if (messages && messages.length) {
        const ul = document.createElement('ul');
        messages.forEach(m => {
            const li = document.createElement('li');
            li.textContent = m;
            ul.appendChild(li);
        });
        box.appendChild(ul);
    }
    formStatus.innerHTML = '';
    formStatus.appendChild(box);
}

function clearFieldErrors() {
    document.querySelectorAll('.field.has-error').forEach(f =>
        f.classList.remove('has-error'));
}

function markFieldErrors(details) {
    // Map a message back onto a field. Backend messages start with the
    // quoted field name ("'marks' must be ..."), but the client-side empty
    // feedback message names the field in prose, so match both forms.
    const names = Object.keys(SLIDERS).concat(['feedback']);
    details.forEach(detail => {
        const lower = detail.toLowerCase();
        const hit = names.find(n =>
            detail.indexOf("'" + n + "'") === 0 || lower.indexOf(n) !== -1);
        if (!hit) return;
        const el = document.getElementById(hit);
        if (el && el.closest('.field')) {
            el.closest('.field').classList.add('has-error');
        }
    });
}

function setLoading(loading) {
    submitBtn.classList.toggle('is-loading', loading);
    submitBtn.disabled = loading;
    submitBtn.querySelector('.btn-text').textContent = loading
        ? 'Analyzing…'
        : 'Run the demo';
}

function hideResults() {
    resultSection.hidden = true;
    errorPanel.hidden = true;
}

function showError(title, message, details) {
    resultSection.hidden = true;
    document.getElementById('errorTitle').textContent = title;
    document.getElementById('errorMessage').textContent = message;

    const list = document.getElementById('errorDetails');
    list.innerHTML = '';
    (details || []).forEach(d => {
        const li = document.createElement('li');
        li.textContent = d;
        list.appendChild(li);
    });
    errorPanel.hidden = false;
    errorPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ---------------------------------------------------------------------
   Client-side pre-check.
   This is a convenience only — the backend validates independently and
   is the authority. It exists to give fast feedback, not to be trusted.
   --------------------------------------------------------------------- */
function preValidate() {
    const problems = [];
    const text = feedbackEl.value.trim();

    if (text.length === 0) {
        problems.push('Please provide some feedback before submitting.');
    } else if (text.length > 2000) {
        problems.push('Feedback must be at most 2000 characters.');
    }
    return problems;
}

/* ---------------------------------------------------------------------
   Branch breakdown
   --------------------------------------------------------------------- */
/* Renders one numeric cell: the value as text, plus a decorative bar.
   Values are written with textContent, never innerHTML. */
function branchCell(value, modifier) {
    const cell = document.createElement('div');
    cell.className = 'branch-cell';

    const num = document.createElement('span');
    num.className = 'branch-value';
    num.textContent = typeof value === 'number' ? value.toFixed(2) : '—';
    cell.appendChild(num);

    const track = document.createElement('div');
    track.className = 'branch-track';
    track.setAttribute('aria-hidden', 'true');

    const fill = document.createElement('div');
    fill.className = 'branch-fill ' + modifier;
    const pct = Math.max(0, Math.min(100, (value || 0) * 100));
    fill.style.width = pct.toFixed(1) + '%';
    track.appendChild(fill);
    cell.appendChild(track);

    return cell;
}

function renderBranchScores(branch, blended) {
    const grid = document.getElementById('branchGrid');
    if (!grid) return;
    grid.innerHTML = '';

    const numeric = (branch && branch.numeric) || {};
    const text = (branch && branch.text) || {};
    const blend = blended || {};

    ['Pattern', 'Numeric branch', 'Text branch', 'Blended'].forEach(function (head) {
        const cell = document.createElement('div');
        cell.className = 'branch-cell branch-head';
        cell.textContent = head;
        grid.appendChild(cell);
    });

    /* Track the widest gap between the two branches, so the note can say
       whether this particular result rested on agreement or on a coin toss. */
    let widestGap = 0;
    BRANCH_ROWS.forEach(function (row) {
        const label = document.createElement('div');
        label.className = 'branch-cell branch-label';
        label.textContent = row[1];
        grid.appendChild(label);
        grid.appendChild(branchCell(numeric[row[0]], 'fill-numeric'));
        grid.appendChild(branchCell(text[row[0]], 'fill-text'));
        grid.appendChild(branchCell(blend[row[0]], 'fill-blend'));

        const gap = Math.abs((numeric[row[0]] || 0) - (text[row[0]] || 0));
        if (gap > widestGap) widestGap = gap;
    });

    const note = document.getElementById('branchNote');
    if (note) {
        note.textContent = widestGap >= 0.35
            ? 'The two branches disagreed sharply on at least one pattern, so this '
            + 'result leans on a blend the model has no strong reason to prefer.'
            : 'The two branches broadly agreed here. Agreement is not correctness — '
            + 'this model is still wrong about one classification in three.';
    }
}

/* ---------------------------------------------------------------------
   Rendering the result
   --------------------------------------------------------------------- */
function setConfidenceBar(id, value) {
    const pct = Math.max(0, Math.min(100, (value || 0) * 100));
    document.getElementById(id).style.width = pct.toFixed(1) + '%';
}

function renderResult(data) {
    /* --- Badge ---------------------------------------------------- */
    const badge = document.getElementById('riskBadge');
    const badgeValue = document.getElementById('riskBadgeValue');
    const level = data.risk_level;

    badge.className = 'risk-badge ' + (
        level === 'HIGH' ? 'risk-high'
            : level === 'MEDIUM' ? 'risk-med'
                : 'risk-low');
    badgeValue.textContent = LEVEL_LABELS[level] || level;

    /* --- Support panel ------------------------------------------- */
    const support = data.support || {};
    const supportPanel = document.getElementById('supportPanel');
    document.getElementById('supportHeading').textContent =
        support.heading || 'If you need support';
    document.getElementById('supportBody').textContent = support.body || '';

    const supportList = supportPanel.querySelector('.support-list');
    supportList.innerHTML = '';
    (support.options || []).forEach(line => {
        const li = document.createElement('li');
        li.textContent = line;
        supportList.appendChild(li);
    });
    supportPanel.classList.toggle('is-elevated', !!data.is_high_risk);

    /* --- Confidence bars ----------------------------------------- */
    document.getElementById('confidenceNote').textContent =
        data.confidence_note || '';
    const scores = data.class_scores || {};
    setConfidenceBar('probLow', scores.LOW);
    setConfidenceBar('probMed', scores.MEDIUM);
    setConfidenceBar('probHigh', scores.HIGH);

    /* --- Branch breakdown ----------------------------------------- */
    renderBranchScores(data.branch_scores, scores);

    /* --- Per-prediction explanation ------------------------------- */
    const explanation = data.explanation || {};
    const container = document.getElementById('featureImportance');
    container.innerHTML = '';

    (explanation.features || []).forEach(f => {
        const li = document.createElement('li');
        li.className = 'influence-item';

        const name = document.createElement('span');
        name.className = 'influence-name';
        name.textContent = f.label;

        const tag = document.createElement('span');
        const magnitude = Math.abs(f.influence || 0);
        if (magnitude < 0.02) {
            tag.className = 'influence-tag influence-none';
            tag.textContent = 'Little effect';
        } else {
            const dirClass = f.direction === 'increased' ? 'influence-up' : 'influence-down';
            tag.className = 'influence-tag ' + dirClass;
            tag.textContent = (f.direction === 'increased' ? 'Raised ' : 'Lowered ')
                + 'this result';
        }

        const detail = document.createElement('span');
        detail.className = 'influence-detail';
        detail.textContent = 'You entered ' + f.submitted_value
            + '; a typical value in this demo data is ' + f.reference_value + '.';

        li.appendChild(name);
        li.appendChild(tag);
        li.appendChild(detail);
        container.appendChild(li);
    });

    /* --- Recommendations ----------------------------------------- */
    const recList = document.getElementById('recommendationsList');
    recList.innerHTML = '';
    (data.recommendations || []).forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;   // textContent, never innerHTML
        recList.appendChild(li);
    });

    /* --- Show ------------------------------------------------------ */
    errorPanel.hidden = true;
    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ---------------------------------------------------------------------
   Submit
   --------------------------------------------------------------------- */
form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearFieldErrors();

    const problems = preValidate();
    if (problems.length) {
        setStatus('invalid', problems);
        markFieldErrors(problems);
        showError('Check your entries',
            'The demo needs a reflection entry before it can run.',
            problems);
        return;
    }

    setStatus('ready', null);
    hideResults();
    setLoading(true);

    /* Numbers are sent as JSON numbers, not strings. */
    const payload = {
        marks: Number(document.getElementById('marks').value),
        attendance: Number(document.getElementById('attendance').value),
        sleep_hours: Number(document.getElementById('sleep_hours').value),
        screen_time: Number(document.getElementById('screen_time').value),
        assignment_delay: Number(document.getElementById('assignment_delay').value),
        feedback: feedbackEl.value.trim(),
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        let data = null;
        try {
            data = await response.json();
        } catch (parseError) {
            data = null;
        }

        if (response.ok && data) {
            renderResult(data);
        } else if (response.status === 400) {
            const details = (data && data.details) || [];
            setStatus('invalid', details);
            markFieldErrors(details);
            showError('Invalid input',
                (data && data.error) || 'The demo rejected these values.',
                details);
        } else if (response.status >= 500) {
            showError('Server unavailable',
                (data && data.error)
                    || 'The demo could not process this request. Please try again.',
                []);
        } else {
            showError('Unexpected response',
                'The demo returned an unexpected response.',
                []);
        }
    } catch (networkError) {
        showError('Cannot reach the demo',
            'The server did not respond. Check that it is running and try again.',
            []);
    } finally {
        setLoading(false);
    }
});

/* ---------------------------------------------------------------------
   Reset / retry
   --------------------------------------------------------------------- */
resetBtn.addEventListener('click', () => {
    hideResults();
    clearFieldErrors();
    setStatus(null, null);
    form.scrollIntoView({ behavior: 'smooth', block: 'start' });
});

retryBtn.addEventListener('click', () => {
    hideResults();
    feedbackEl.focus();
});

/* ---------------------------------------------------------------------
   Printing
   --------------------------------------------------------------------- */
/* The privacy + limitations panel is collapsed by default. Open it before
   printing so a printed page is never silently missing every caveat. */
let printRestore = null;

window.addEventListener('beforeprint', () => {
    const panels = Array.from(document.querySelectorAll('details.disclosure'));
    if (!panels.length) return;
    printRestore = panels.map(p => p.open);
    panels.forEach(p => { p.open = true; });
});

window.addEventListener('afterprint', () => {
    if (!printRestore) return;
    const panels = Array.from(document.querySelectorAll('details.disclosure'));
    panels.forEach((p, i) => { p.open = printRestore[i]; });
    printRestore = null;
});