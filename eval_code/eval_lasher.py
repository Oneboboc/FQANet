from rgbt import LasHeR

lasher = LasHeR()

"""
LasHeR have 3 benchmarks: PR, NPR, SR
"""

# Register your tracker
lasher(
    tracker_name="tracker2",
    result_path="",     
    bbox_type="ltwh")

# Evaluate multiple trackers
pr_dict = lasher.PR()
npr_dict = lasher.NPR()
sr_dict = lasher.SR()

print(pr_dict["tracker1"][0])
print(npr_dict["tracker1"][0])
print(sr_dict["tracker1"][0])

lasher.draw_plot(metric_fun=lasher.PR)
lasher.draw_plot(metric_fun=lasher.NPR)
lasher.draw_plot(metric_fun=lasher.SR)