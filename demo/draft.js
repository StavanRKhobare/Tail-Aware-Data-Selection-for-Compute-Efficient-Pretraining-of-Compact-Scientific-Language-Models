const byId = id => document.getElementById(id);
const keys = ['title', 'problem', 'method', 'results', 'limitations'];
const status = byId('draft_status');
function countWords() {
  const text = byId('output_abstract').value.trim();
  byId('draft_count').textContent = `${text ? text.split(/\s+/).length : 0} words`;
}
byId('output_abstract').addEventListener('input', countWords);
const engineSelect = byId('draft_mode');
engineSelect.addEventListener('change', () => {
  if (!['auto', 'formatter'].includes(engineSelect.value)) {
    byId('draft_fidelity').value = 'review';
  }
});
fetch('/api/draft/status').then(response => response.json()).then(data => {
  const availability = [];
  for (const [id, engine] of Object.entries(data.engines)) {
    let option = engineSelect.querySelector(`option[value="${id}"]`);
    if (!option) option = engineSelect.appendChild(new Option(engine.label, id));
    option.disabled = !engine.available;
    option.textContent = engine.label + (engine.available ? '' : ' (unavailable)');
    availability.push(`${engine.label}: ${engine.available ? 'ready' : engine.reason}`);
  }
  byId('draft_model_status').textContent = availability.join(' / ');
  if (engineSelect.selectedOptions[0].disabled) engineSelect.value = 'formatter';
}).catch(() => { byId('draft_model_status').textContent = 'Could not check model availability.'; });
const examples = {
  tail_study: {
    title: 'Tail-Aware Data Selection for Compact Scientific Language Models',
    problem: 'Scientific corpora contain many papers from popular fields and fewer papers on specialized topics. This imbalance can leave a small language model with limited exposure to rare topics during pretraining.',
    method: 'Compare random and tail-aware data selection for continued pretraining of SmolLM2-135M. Each run starts from the same base checkpoint, uses seed 42, and processes approximately 60 million training tokens. Measure next-token loss overall and separately for head, middle, and tail topics on the same validation split.',
    results: 'Tail-aware selection lowered tail validation loss from 3.0216 to 2.9892. Overall loss rose from 2.8568 to 2.8691, and head loss rose from 2.8383 to 2.8596.',
    limitations: 'The continued-pretraining comparison uses one seed and one scientific corpus. The evaluation measures prediction loss on held-out text. Drafting quality requires a separate evaluation.'
  },
  noisy_classifier: {
    title: 'Document Classification with Noisy Headings',
    problem: 'Evaluate how missing and misspelled headings affect document classification. Headings provide useful cues, and scans can lose them or introduce recognition errors.',
    method: 'Use 800 training and 200 validation documents from four categories. Compare a word-based classifier with a character-based classifier on clean text and on copies with corrupted headings. Use the same validation documents for both models.',
    results: 'On clean text, the word-based classifier reached 84% accuracy and the character-based classifier reached 82%. Under heading corruption, their accuracies were 71% and 78%, respectively. The character-based model had lower accuracy on clean inputs and higher accuracy on corrupted inputs.',
    limitations: 'The corruption was simulated. The study used one collection of documents and did not measure how either model handles errors from a production scanner.'
  },
  forecast_plan: {
    title: 'Weather-Aware Electricity Demand Forecasting',
    problem: 'Electricity demand can change rapidly during hot weather. A daily forecasting model should capture these changes while accounting for recurring patterns such as weekdays and holidays.',
    method: 'Compare a seasonal baseline with a forecasting model that uses lagged demand, temperature, humidity, and calendar features. Split the data chronologically and evaluate both models on the same forecast dates. Report errors separately for ordinary days and the hottest days in the evaluation period.',
    results: '',
    limitations: 'The proposed evaluation covers a single service area. Weather observations available after the forecast cutoff must be excluded. Performance in another region will require a separate evaluation.'
  },
  numbers: {title:'Classifier Evaluation',problem:'Evaluate a classifier.',method:'Use 800 training and 200 validation examples.',results:'Validation accuracy was 81.2%.',limitations:''},
  negative: {title:'Rare-topic Selection Study',problem:'Rare categories receive little training data.',method:'Train a 17.1M parameter model for 60M tokens.',results:'Tail-aware selection did not improve rare-topic loss.',limitations:''},
  positive: {title:'Rare-topic Selection Study',problem:'Rare categories receive little training data.',method:'Train a 17.1M parameter model for 60M tokens.',results:'Tail-aware selection improved rare-topic loss.',limitations:''},
  missing: {title:'Logistic Regression Study',problem:'',method:'Use logistic regression with 800 training examples.',results:'',limitations:''}
};
byId('draft_example').onclick = () => {
  const example = examples[byId('draft_example_choice').value];
  keys.forEach(key => { byId('draft_' + key).value = example[key]; });
  status.textContent = 'Example loaded. Review the notes or create a draft.';
};
byId('draft_form').onsubmit = async event => {
  event.preventDefault();
  const button = byId('draft_run');
  button.disabled = true;
  status.textContent = 'Preparing draft...';
  byId('draft_warnings').replaceChildren();
  const payload = Object.fromEntries(keys.map(key => [key, byId('draft_' + key).value]));
  payload.fidelity_mode = byId('draft_fidelity').value;
  payload.mode = byId('draft_mode').value;
  payload.title_mode = byId('draft_title_mode').value;
  payload.lock_results = byId('draft_lock_results').value === 'true';
  payload.polish = byId('draft_polish').value === 'true';
  payload.max_tokens = Number(byId('draft_max_tokens').value);
  try {
    const response = await fetch('/api/draft', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data = await response.json();
    byId('draft_raw').textContent = data.raw_output ?? (data.model_generated ? data.raw_abstract : '');
    byId('draft_raw_panel').hidden = !byId('draft_raw').textContent;
    if (!response.ok && data.raw_output) byId('draft_raw_panel').open = true;
    if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Check the drafting settings.');
    byId('output_title').value = data.title;
    byId('output_abstract').value = data.abstract;
    byId('draft_origin').textContent = data.abstract_origin;
    byId('draft_raw_panel').open = data.model_generated && payload.lock_results && Boolean(payload.results.trim());
    status.textContent = `${data.notice} ${data.editing}. Automatic checks do not verify all claims.`;
    for (const warning of data.warnings) {
      const item = document.createElement('li');
      item.textContent = warning;
      byId('draft_warnings').append(item);
    }
    countWords();
  } catch (error) { status.textContent = error.message + ' Previous draft retained.'; }
  finally { button.disabled = false; }
};
const draftText = () => byId('output_title').value + '\n\n' + byId('output_abstract').value;
byId('draft_copy').onclick = async () => {
  try { await navigator.clipboard.writeText(draftText()); status.textContent = 'Draft copied.'; }
  catch { status.textContent = 'Copy unavailable. Select the draft and copy it manually.'; }
};
byId('draft_export').onclick = () => {
  const url = URL.createObjectURL(new Blob([draftText()], {type:'text/plain;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url; link.download = 'research-draft.txt'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  status.textContent = 'Draft exported.';
};
