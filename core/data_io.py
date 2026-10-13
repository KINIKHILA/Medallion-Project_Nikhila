import re
from pathlib import Path
from typing import Any, Dict, List, Union

FilePath = Union[str, Path]


def read_tabular_datasets(file_path: FilePath) -> List[Dict[str, Any]]:
	"""Read a CSV or expand every worksheet in an XLSX into named datasets."""
	import pandas as pd

	path = Path(file_path)
	suffix = path.suffix.lower()
	if suffix == ".csv":
		return [
			{
				"path": str(path),
				"table_name": path.stem,
				"sheet_name": None,
				"dataframe": pd.read_csv(path),
			}
		]
	if suffix != ".xlsx":
		raise ValueError(f"Unsupported input file type '{suffix}'. Use CSV or XLSX.")

	sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
	if not sheets:
		raise ValueError(f"Excel workbook '{path.name}' contains no worksheets.")

	datasets = []
    # Keep the workbook and sheet identity together for STTM matching and outputs.
	used_names = set()
	for sheet_name, dataframe in sheets.items():
		safe_sheet_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sheet_name)).strip("_")
		safe_sheet_name = safe_sheet_name or "sheet"
		table_name = f"{path.stem}_{safe_sheet_name}"
		base_name = table_name
		counter = 2
		while table_name.casefold() in used_names:
			table_name = f"{base_name}_{counter}"
			counter += 1
		used_names.add(table_name.casefold())
		datasets.append(
			{
				"path": str(path),
				"table_name": table_name,
				"sheet_name": str(sheet_name),
				"dataframe": dataframe,
			}
		)
	return datasets