const fields = ['overall', 'head', 'mid', 'tail'];
const showLoss = value => Number.isFinite(value) ? value.toFixed(4) : '—';
const delta = (tail, random) => fields.map(field =>
  Number.isFinite(tail[field]) && Number.isFinite(random[field]) ? tail[field] - random[field] : null
);
const values = row => fields.map(field => row[field]);

function row(body, label, cells, format = showLoss) {
  const line = body.insertRow();
  const heading = document.createElement('th');
  heading.scope = 'row';
  heading.textContent = label;
  line.append(heading);
  cells.forEach(value => { line.insertCell().textContent = format(value); });
}

function renderLoss(data) {
  const body = document.getElementById('compare_losses');
  if (data.scratch_34m) {
    const run = data.scratch_34m;
    document.getElementById('compare_34m').textContent =
      `34M scratch baseline: ${showLoss(run.final_val_loss)} overall validation loss after ${run.tokens_seen.toLocaleString('en-US')} training tokens with random selection. This run uses a larger training budget and has no paired tail-aware run.`;
  }
  const scratch = Object.fromEntries(data.scratch.map(item => [item.model, item]));
  if (scratch.random && scratch.tail_aware) {
    row(body, '17M scratch · random', values(scratch.random));
    row(body, '17M scratch · tail-aware', values(scratch.tail_aware));
    row(body, '17M scratch · tail-aware − random', delta(scratch.tail_aware, scratch.random));
  }
  if (!data.smollm2) return;
  const runs = data.smollm2.runs;
  const loss = run => ({ overall: run.overall, ...run.loss_by_group });
  row(body, 'SmolLM2 · base', values(loss(runs.base)));
  row(body, 'SmolLM2 · random', values(loss(runs.random)));
  row(body, 'SmolLM2 · tail-aware', values(loss(runs.tail_aware)));
  const differences = delta(loss(runs.tail_aware), loss(runs.random));
  row(body, 'SmolLM2 · tail-aware − random', differences);
  const tail = differences[3];
  document.getElementById('compare_research_status').textContent = tail === null
    ? 'SmolLM2 tail loss is unavailable.'
    : tail === 0
      ? 'Both SmolLM2 selection strategies matched on tail loss at seed 42.'
      : `SmolLM2 tail-aware selection ${tail < 0 ? 'lowered' : 'raised'} tail loss by ${Math.abs(tail).toFixed(4)} at seed 42.`;
}

function renderDrafts(data) {
  if (!data) return;
  const select = document.getElementById('compare_case');
  const inputs = document.getElementById('compare_input');
  const checks = document.getElementById('compare_checks');
  const outputs = document.getElementById('compare_outputs');
  const summary = document.getElementById('compare_formats');
  document.getElementById('compare_review').textContent = data.review || '';
  for (const [id, label] of Object.entries(data.engines)) {
    const successes = data.cases.filter(item => item.outputs[id]?.status === 200).length;
    row(summary, label, [`${successes} / ${data.cases.length}`], String);
  }
  for (const [index, item] of data.cases.entries()) {
    select.add(new Option(item.input.id.replaceAll('_', ' '), index));
  }
  const renderCase = () => {
    const item = data.cases[Number(select.value)];
    inputs.textContent = ['title', 'problem', 'method', 'results', 'limitations']
      .filter(field => item.input[field])
      .map(field => `${field[0].toUpperCase() + field.slice(1)}: ${item.input[field]}`).join('\n');
    checks.textContent = item.input.checks?.join(' ') || '';
    outputs.replaceChildren();
    for (const [id, label] of Object.entries(data.engines)) {
      const result = item.outputs[id];
      const card = document.createElement('article');
      card.className = 'card';
      const heading = document.createElement('h3');
      heading.textContent = label;
      const content = document.createElement('p');
      content.className = 'result';
      content.textContent = result.status === 200
        ? [result.title, result.raw_abstract].filter(Boolean).join('\n\n')
        : [result.detail, result.raw_output].filter(Boolean).join('\n\n');
      card.append(heading, content);
      if (result.warnings?.length) {
        const warnings = document.createElement('p');
        warnings.className = 'note';
        warnings.textContent = result.warnings.join(' ');
        card.append(warnings);
      }
      outputs.append(card);
    }
  };
  document.getElementById('compare_draft_status').hidden = true;
  document.getElementById('compare_draft_data').hidden = false;
  select.addEventListener('change', renderCase);
  renderCase();
}

fetch('/api/comparisons').then(response => {
  if (!response.ok) throw Error('Could not load comparison results.');
  return response.json();
}).then(data => {
  renderLoss(data);
  renderDrafts(data.drafts);
}).catch(() => {
  document.getElementById('compare_research_status').textContent = 'Could not load comparison results.';
});
