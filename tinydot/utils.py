def flatten(array):
  res = []
  for item in array:
    if isinstance(item, (tuple, list)):
      for subitem in flatten(item):
        res.append(subitem)
    else:
      res.append(item)
  return res

