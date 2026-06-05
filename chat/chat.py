from embeddings.similarity import getSimilarProducts


def chatter(text):
    result = getSimilarProducts(text)
    return result
    
    