import math
def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    # Write code here
    encoded = []
    for v in values:
        theta = (2 * math.pi * v) / period
        encoded.append([math.sin(theta), math.cos(theta)])
    return encoded
    pass 
