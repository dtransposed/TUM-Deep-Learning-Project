'''! party07 !'''
# Here we shall gather the different pooling strategies that we apply

def average_pooling(img_features):
    # simple average pooling.
    # averages over all frames of a video
    
    img_mean = img_features.mean(0)
    
    return img_mean.view(-1)