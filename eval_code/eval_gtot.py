from rgbt import GTOT
from rgbt.utils import RGBT_start,RGBT_end

RGBT_start()
gtot = GTOT()

# Register your tracker
gtot(
    tracker_name="TQRT",    
    result_path="", 
    bbox_type="ltwh",
    prefix="")



# Evaluate multiple trackers
pr_dict = gtot.MPR()
print(pr_dict["TQRT"][0])

# Evaluate single tracker
jmmac_sr,_ = gtot.MSR("TQRT")
print("TQRT MSR:\t", jmmac_sr)

# Draw a curve plot
gtot.draw_plot(metric_fun=gtot.MPR)
gtot.draw_plot(metric_fun=gtot.MSR)

RGBT_end()