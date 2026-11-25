"""Transparent editing and review checks for Phase 1 research drafts."""
import re


def edit_abstract(text, polish=False):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    unique = []
    seen = set()
    for sentence in sentences:
        key = ' '.join(sentence.lower().split())
        if key and key not in seen:
            seen.add(key)
            unique.append(sentence)
    text = ' '.join(unique)
    if polish:
        # Anchored substitutions retain numbers, predicates, and negation.
        replacements = [
            (r'(^|(?<=[.!?]) )Use ', r'\1The study uses '),
            (r'(^|(?<=[.!?]) )Compare ', r'\1The study compares '),
            (r'(^|(?<=[.!?]) )Train ', r'\1The study trains '),
            (r'(^|(?<=[.!?]) )Evaluate ', r'\1The study evaluates '),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
    return text


def compose_abstract(notes):
    """Combine a simple objective and method while preserving supplied claims."""
    problem, method = (notes.get(field, '').strip() for field in ('problem', 'method'))
    objective = re.fullmatch(r'(Evaluate|Compare|Assess|Investigate) ([^.!?\n]+)\.?', problem)
    procedure = re.match(r'(Use|Train|Compare|Evaluate) (.+)', method, re.DOTALL)
    if objective and procedure:
        verb = {'Use': 'uses', 'Train': 'trains', 'Compare': 'compares', 'Evaluate': 'evaluates'}
        method = f'To {objective[1].lower()} {objective[2]}, the study {verb[procedure[1]]} {procedure[2]}'
        problem = ''

    parts = (edit_abstract(problem, polish=True), edit_abstract(method, polish=True),
             notes.get('results', ''), notes.get('limitations', ''))
    sentences = [part if part.endswith(('.', '!', '?')) else part + '.' for part in parts if part]
    return edit_abstract(' '.join(sentences))


def mostly_copies_notes(notes, text):
    def sentences(value):
        return [tuple(re.findall(r'\w+', sentence.casefold())) for sentence in re.split(r'(?<=[.!?])\s+|\n+', value) if sentence.strip()]

    supplied = {sentence for value in notes.values() for sentence in sentences(value)}
    generated = sentences(text)
    total_words = sum(map(len, generated))
    copied_words = sum(len(sentence) for sentence in generated if sentence in supplied)
    return total_words > 0 and copied_words / total_words >= 0.8


def review_notes(notes, output):
    source = ' '.join(notes.values())
    numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?', source))
    generated = set(re.findall(r'\b\d+(?:\.\d+)?%?', output))
    warnings = []
    if generated - numbers:
        warnings.append('Output contains numerical values absent from the notes: ' + ', '.join(sorted(generated - numbers)))
    if numbers - generated:
        warnings.append('Check omitted numerical values: ' + ', '.join(sorted(numbers - generated)))
    if not notes.get('results', '').strip():
        warnings.append('Results were not supplied. Check that the draft claims no findings.')
    if re.search(r'\b(no|not|without|failed)\b', source, re.I) and not re.search(r'\b(no|not|without|failed)\b', output, re.I):
        warnings.append('Check whether a negative or qualified statement was lost.')
    return warnings


def has_repetition(text):
    sentences = [sentence.casefold() for sentence in re.split(r'(?<=[.!?])\s+', text.strip())]
    if any(left == right for left, right in zip(sentences, sentences[1:])):
        return True
    words = text.lower().split()
    for width in (3, 4, 5):
        phrases = [tuple(words[i:i + width]) for i in range(len(words) - width + 1)]
        if phrases and max(phrases.count(phrase) for phrase in set(phrases)) >= 4:
            return True
    return False
