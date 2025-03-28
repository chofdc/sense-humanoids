from pathlib import Path
import os
from force_scripts import online_force_estimation, online_paper_plot


# Make sure the "DATA" folder exists, otherwise create it
if not os.path.exists("DATA"):
    os.makedirs("DATA")

online_paper_plot.main()
online_force_estimation.main()
