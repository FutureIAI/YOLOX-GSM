import torch

def kl_divergence(p, q):
    # 防止除以零，使用一个很小的值进行平滑
    # p = p + 1e-10
    # q = q + 1e-10
    kl_div = torch.sum(p * torch.log(p / q))
    return kl_div


def cosine_similarity(vec1, vec2):
    # 计算内积和向量的模长
    dot_product = torch.sum(vec1 * vec2)
    norm_vec1 = torch.sqrt(torch.sum(vec1 * vec1))
    norm_vec2 = torch.sqrt(torch.sum(vec2 * vec2))
    cos_sim = dot_product / (norm_vec1 * norm_vec2)
    return cos_sim