import os
from force_scripts import online_walking_force_estimation #, online_walking_paper_plot


# Make sure the "DATA" folder exists, otherwise create it
if not os.path.exists("DATA"):
    os.makedirs("DATA")
    
online_walking_force_estimation.main()
#online_walking_paper_plot.main()