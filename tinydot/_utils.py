def _flatten(array):
  res = []
  for item in array:
    if isinstance(item, (tuple, list)):
      for subitem in _flatten(item):
        res.append(subitem)
    else:
      res.append(item)
  return res

def _get_index(coord, shape):
  mul = 1
  index = 0
  for i in reversed(range(len(shape))):
    index += mul * coord[i]
    mul *= shape[i]
  return index

def _reshape(array, shape):
  if len(shape) == 1:
    return array

  span = 1
  for i in shape[1:]:
    span *= i

  return [
      _reshape(array[offset:offset + span], shape[1:])
      for offset in range(0, len(array), span)
  ]
