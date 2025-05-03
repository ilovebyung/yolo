import datetime

def get_date_in_yyyymmdd():
  """Returns the current date in YYYYMMDD format."""
  now = datetime.datetime.now()
  return now.strftime("%Y%m%d")

def format_date(date):
  """Returns the current date in YYYYMMDD format."""
  return date.strftime("%Y%m%d")

def get_time_in_mmddss():
  """Returns the current time in mmddss format."""
  now = datetime.datetime.now()
  return now.strftime("%H%M%S")

def format_time(time):
  """Returns the current time in mmddss format."""
  now = datetime.datetime.now()
  return time.strftime("%H%M%S")
  
def get_filename():
  """Returns the current date in YYYYMMDD_mmddss format."""
  now = datetime.datetime.now()
  filename = now.strftime("%Y%m%d") + '_' +  now.strftime("%H%M%S") + '.png'
  return filename

if __name__ == "__main__":
    print(get_date_in_yyyymmdd())
    print(get_time_in_mmddss())  
    print(get_filename())
    filename = get_filename()  

