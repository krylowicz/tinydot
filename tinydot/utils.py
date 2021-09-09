def flatten(array):
  res = []
  for item in array:
    if isinstance(item, (tuple, list)):
      for subitem in flatten(item):
        res.append(subitem)
    else:
      res.append(item)
  return res

def get_value_at_index(coord, size):
  mul = 1
  index = 0
  for i in range(len(size)):
    index += mul * coord[i]
    mul *= size[i]
  return index

