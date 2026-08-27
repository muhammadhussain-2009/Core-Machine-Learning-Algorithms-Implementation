# The log1p transformation compresses the range of nonnegative data while handling zero naturally.
import math
def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # Write code here
    y = [math.log(1+v) for v in values]
    return y 
    pass
    
