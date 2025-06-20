import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


from .catsample import sample_categorical

def get_graph(config):
    if config.graph.type == "uniform":
        return Uniform(config.graph)
    elif config.graph.type == "absorb":
        return Absorbing(config.graph)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass


    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass


    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass


    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.select_dim).to(rate) + rate)

    
    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass
    

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass


    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """
    def __init__(self, dims):
        self._xdim = dims.x_dim
        self._edim = dims.e_dim

    def select_dim(self, data):
        if data.ndim == 2:
            return self._xdim
        elif data.ndim ==3:
            return self._edim
        else:
            raise ValueError(f"Unsupported  tensor shape.")

    
    @property
    def absorb(self):
        return False

    def rate(self, i):
        dim = self.select_dim(i)
        edge = torch.ones(*i.shape, edge, device=i.device) / edge
        edge = edge.scatter(-1, i[..., None], - (edge - 1) / edge)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        dim = self.select_dim(i)
        trans = torch.ones(*i.shape, dim, device=i.device) * (1 - (-sigma).exp()) / dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans
    
    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        dim = self.select_dim(i)
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        if score.ndim == 3:
            epow = (-dsigma).exp()[..., None]
        elif score.ndim == 4:
            epow = (-dsigma).exp()[..., None, None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, batch_dims, sample_type):
        if sample_type == 'node':
            return torch.randint(0, self._xdim, batch_dims)
        elif sample_type == 'edge':
            return torch.randint(0, self._edim, batch_dims)


    def score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        dim = score.shape[-1]
        ratio = 1 - dim / (esigm1 + dim)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (dim - 1) / dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (dim - 2)) / dim 
        )

        #positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / dim
        return pos_term - neg_term + const


class Absorbing(Graph):
    def __init__(self, dims):
        super().__init__()
        self._xdim = dims.x_dim
        self._edim = dims.e_dim

    def select_dim(self, data):
        if data.ndim == 2:
            return self._xdim + 1
        elif data.ndim ==3:
            return self._edim + 1
        else:
            raise ValueError(f"Unsupported  tensor shape.")
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.select_dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        dim = self.select_dim(i)
        return F.one_hot((dim - 1) * torch.ones_like(i), num_classes=dim) - F.one_hot(i, num_classes=dim)        

    def transp_rate(self, i):
        dim = self.select_dim(i)
        edge = -F.one_hot(i, num_classes=dim)
        edge[i == dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma):
        dim = self.select_dim(i)
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=dim)
        edge += torch.where(
            i == dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        dim = self.select_dim(i)
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, dim - 1, i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        if score.ndim == 3:
            extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
            score *= dsigma.exp()[:, None]
        elif score.ndim == 4:
            extra_const = (1 - (dsigma).exp()[..., None]) * score.sum(dim=-1)
            score *= dsigma.exp()[:, None, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims, sample_type):
        if sample_type == 'node':
            return (self._xdim) * torch.ones(*batch_dims, dtype=torch.int64)
        elif sample_type == 'edge':
            return (self._edim) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        dim = score.shape[-1] 

        rel_ind = x == dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy
    