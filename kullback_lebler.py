from numpy import log

def kld(model1,model2):
    vocabulary = model1.lm.keys()
    div = 0
    for v in vocabulary:
        div += model1.getProbSm(v)*(log(model1.getProbSm(v))-log(model2.getProbSm(v)))
    return div
