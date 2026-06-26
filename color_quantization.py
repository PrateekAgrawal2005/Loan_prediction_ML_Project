import numpy as np
from sklearn.cluster import KMeans

def quantify_image_colors(image, n_clusters):
    image = np.array(image)

    h, w, c = image.shape
    image_2d = image.reshape(h * w, c)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = model.fit_predict(image_2d)

    rgb_codes = model.cluster_centers_.round(0).astype(int)
    quantized_image = np.reshape(rgb_codes[labels], (h, w, c))

    return image, quantized_image