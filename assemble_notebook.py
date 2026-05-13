"""Assemble all parts into the final notebook by importing each module."""
import sys, importlib, types, nbformat

def load_cells(filename):
    """Execute a part script and return its `cells` list."""
    spec = importlib.util.spec_from_file_location("_part", filename)
    mod  = importlib.util.module_from_spec(spec)
    # Provide nbformat in module namespace
    mod.nbformat = nbformat
    spec.loader.exec_module(mod)
    return mod.cells

all_cells = []
for i in range(1, 7):
    c = load_cells(f"build_notebook_part{i}.py")
    all_cells.extend(c)
    print(f"Part {i}: +{len(c)} cells  (total so far: {len(all_cells)})")

nb = nbformat.v4.new_notebook()
nb.cells = all_cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}
nb.metadata["language_info"] = {"name": "python", "version": "3.10.0"}

out = "Silao_Failure_Forecast_v6.ipynb"
nbformat.write(nb, out)
print(f"\n✅ Notebook escrito: {out}  ({len(all_cells)} celdas)")
