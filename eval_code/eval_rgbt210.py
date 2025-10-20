from rgbt import RGBT210

rgbt210 = RGBT210()

# Register your tracker
rgbt210(
    tracker_name="TQRT",        
    result_path="", 
    bbox_type="ltwh",
    prefix="")

# Evaluate multiple trackers

sr_dict = rgbt210.SR()
print(sr_dict["TQRT"][0])

pr_dict = rgbt210.PR()
print(pr_dict["TQRT"][0])

rgbt210.draw_plot(metric_fun=rgbt210.PR)
