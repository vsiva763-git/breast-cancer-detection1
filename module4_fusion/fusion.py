def weighted_fusion(image_prob, tabular_prob, nlp_prob,
                    w_image=0.45, w_tabular=0.40, w_nlp=0.15):
    """
    Combine 3 model probabilities into one fused probability.
    Weights reflect model reliability:
      - Image CNN   : 45% (trained on real histopathology)
      - Tabular ML  : 40% (clinical features, high AUC)
      - NLP         : 15% (synthetic data, supporting signal)
    """
    # Validate inputs
    for name, val in [('image', image_prob), ('tabular', tabular_prob), ('nlp', nlp_prob)]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} probability must be between 0 and 1, got {val}")

    fused = (w_image * image_prob +
             w_tabular * tabular_prob +
             w_nlp * nlp_prob)

    return round(float(fused), 4)


def confidence_weighted_fusion(image_prob, tabular_prob, nlp_prob):
    """
    Advanced fusion: weight each module by how confident it is.
    More confident predictions get higher weight automatically.
    """
    def confidence(prob):
        # Distance from 0.5 = how confident the model is
        return abs(prob - 0.5) * 2  # scale to 0-1

    c_image   = confidence(image_prob)
    c_tabular = confidence(tabular_prob)
    c_nlp     = confidence(nlp_prob)

    total = c_image + c_tabular + c_nlp

    if total == 0:
        # All models uncertain — fall back to equal weights
        return round((image_prob + tabular_prob + nlp_prob) / 3, 4)

    w_image   = c_image / total
    w_tabular = c_tabular / total
    w_nlp     = c_nlp / total

    fused = (w_image * image_prob +
             w_tabular * tabular_prob +
             w_nlp * nlp_prob)

    return round(float(fused), 4)
