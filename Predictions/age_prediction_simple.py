from MarathonAgeModel import MarathonModel
from age_reference_builder import to_hhmmss
from logging_setup import easy_logging
import pandas as pd

easy_logging(True)

model = MarathonModel(
    data_path=r"C:\Users\andre\github\Marathons\Data",
    verbose=True,
)
model.run()

result = model.predict_with_uncertainty(
    race_id="Белые ночи",
    gender="M",
    age=40.0,
    year=2025,
    confidence=0.95,
    method="analytical",
)

time_pred_hhmmss = to_hhmmss(result["time_pred"])
time_lower_hhmmss = to_hhmmss(result["time_lower"])
time_upper_hhmmss = to_hhmmss(result["time_upper"])

print(f"Predicted time: {time_pred_hhmmss}")
print(f"95%: [{time_lower_hhmmss}, {time_upper_hhmmss}]")
