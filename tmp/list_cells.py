import nbformat

with open(r'd:\Lab\BDA\Project\notebooks\02_text_classification_models.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

with open(r'd:\Lab\BDA\Project\tmp\cells_output.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            out.write(f"--- Code Cell {i} ---\n")
            out.write(cell.source + "\n")
        elif cell.cell_type == 'markdown':
            out.write(f"--- Mkd Cell {i} ---\n")
            out.write(cell.source[:100] + ("..." if len(cell.source) > 100 else "") + "\n")
