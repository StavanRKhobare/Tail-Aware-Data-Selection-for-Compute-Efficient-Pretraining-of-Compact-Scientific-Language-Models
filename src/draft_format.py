"""Shared structured drafting prompt used for training and application inference."""
FIELDS = ('problem', 'method', 'results', 'limitations')


def structured_prompt(title='', **notes):
    lines = ['Write a title and abstract using only the supplied facts.',
             f'Working title: {title.strip()}']
    lines.extend(f'{field.capitalize()}: {notes.get(field, "").strip()}' for field in FIELDS)
    return '\n'.join(lines) + '\nTitle:'
