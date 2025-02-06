import datetime

# If we need to support more date formats, best to use dateutil.parser
# import dateutil

def parse_date(date_str):
    formats = ["%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%m/%d/%y", "%m/%d/%Y"]
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date format for '{date_str}' is not supported")
