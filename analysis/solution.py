# ## Solutions

# %%
import pandas as pd
from utils import NetworkFlowModel
from pathlib import Path
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


solution_path = Path(__file__).parent.joinpath("solutions")
if not solution_path.exists():
    solution_path.mkdir()

input_path = Path(__file__).parent.parent.joinpath("data/NetworkFlowProblem-Data.xlsx")

wb = pd.ExcelFile(input_path)
data = {}
for sheet in wb.sheet_names:
    if "Input" in sheet:
        data[sheet] = wb.parse(sheet)

# %%
for input_name, input_data in data.items():
    logging.info(f"Solving {input_name}")
    model = NetworkFlowModel(input_data)
    solution = model.solve()
    solution.demands.to_csv(
        solution_path.joinpath(f"{input_name}_demands.csv"), index=False
    )
    fig = solution.visualize(ret=True)
    fig.savefig(
        solution_path.joinpath(f"{input_name}_network.png"),
        dpi=300,
        bbox_inches="tight",
    )

# %%
