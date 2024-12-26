def valid(arr, name):
    val = 0
    if name == 'h2':
        val = 30.95703809
    elif name == 'h2o':
        val = 43.90548136
    elif name == 'h':
        val = 18.82442264
    elif name == 'o':
        val = 18.9010205
    elif name == 'oh':
        val = 29.78003733
    elif name == 'o2':
        val = 33.9328105

    return [i - val for i in arr]