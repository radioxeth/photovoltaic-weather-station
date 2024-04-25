import datetime
from utils import get_sunrise_sunset
from datetime import date

# Example usage:
date_input = date(2024, 4, 25)  # Change to desired date
sunrise, sunset = get_sunrise_sunset(
    latitude=39.6943265,
    longitude=-105.0699361,
    elevation=1685,
    date_input=date_input,
)
print("Sunrise:", sunrise)
print("Sunset:", sunset)
# convert sunrise and sunset to datetime objects


sunrise_time = datetime.datetime.strptime(sunrise, "%H:%M:%S")
sunset_time = datetime.datetime.strptime(sunset, "%H:%M:%S")
diff = sunset_time - sunrise_time
# convert diff to seconds
diff = diff.total_seconds()
print("Difference in seconds:", diff)
