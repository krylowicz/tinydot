def flatten(array):
  res = []
  for item in array:
    if isinstance(item, (tuple, list)):
      for subitem in flatten(item):
        res.append(subitem)
    else:
      res.append(item)
  return res

def get_index(coord, shape):
  mul = 1
  index = 0
  for i in reversed(range(len(shape))):
    index += mul * coord[i]
    mul *= shape[i]
  return index

def reshape(array, shape):
  if len(shape) == 1:
    return array

  result = []
  
  span = 1
  for i in shape[1:]:
    span *= i
  
  for offset in range(0, len(array), span):
    result.append(reshape(array[offset:offset+span], shape[1:]))

  return result

