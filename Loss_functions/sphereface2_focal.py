import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SphereFace2FocalLoss(nn.Module):
    """ 
    SphereFace2 with Focal Loss
    reference: <SphereFace2: Binary Classification is All You Need for Deep Face Recognition>
    margin='C' -> SphereFace2-C
    margin='A' -> SphereFace2-A
    margin='M' -> SphereFace2-M
    """
    def __init__(self, feat_dim, num_class, magn_type='C',
                 alpha=0.7, r=40., m=0.4, t=3., lw=50.,
                 focal_gamma=2.0):  # Added focal_gamma parameter
        super().__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.magn_type = magn_type
        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw
        self.focal_gamma = focal_gamma  # Focal loss focusing parameter
        
        # init weights
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)
        
        # init bias
        z = alpha / ((1. - alpha) * (num_class - 1.))
        if magn_type == 'C':
            ay = r * (2. * 0.5**t - 1. - m)
            ai = r * (2. * 0.5**t - 1. + m)
        elif magn_type == 'A':
            theta_y = min(math.pi, math.pi/2. + m)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.)**t - 1.)
            ai = r * (2. * 0.5**t - 1.)
        elif magn_type == 'M':
            theta_y = min(math.pi, m * math.pi/2.)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.)**t - 1.)
            ai = r * (2. * 0.5**t - 1.)
        else:
            raise NotImplementedError
            
        temp = (1. - z)**2 + 4. * z * math.exp(ay - ai)
        b = (math.log(2. * z) - ai - math.log(1. - z + math.sqrt(temp)))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)
        
    def forward(self, x, y):
        # Normalize weights
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)
        
        # Compute cosine similarity
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, y.view(-1, 1), 1.)
        
        # Compute margin-adjusted cosine similarity
        with torch.no_grad():
            if self.magn_type == 'C':
                g_cos_theta = 2. * ((cos_theta + 1.) / 2.).pow(self.t) - 1.
                g_cos_theta = g_cos_theta - self.m * (2. * one_hot - 1.)
            elif self.magn_type == 'A':
                theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1.-1e-5))
                theta_m.scatter_(1, y.view(-1, 1), self.m, reduce='add')
                theta_m.clamp_(1e-5, math.pi)
                g_cos_theta = torch.cos(theta_m)
                g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            elif self.magn_type == 'M':
                m_theta = torch.acos(cos_theta.clamp(-1+1e-5, 1.-1e-5))
                m_theta.scatter_(1, y.view(-1, 1), self.m, reduce='multiply')
                m_theta.clamp_(1e-5, math.pi)
                g_cos_theta = torch.cos(m_theta)
                g_cos_theta = 2. * ((g_cos_theta + 1.) / 2.).pow(self.t) - 1.
            else:
                raise NotImplementedError
            d_theta = g_cos_theta - cos_theta
        
        logits = self.r * (cos_theta + d_theta) + self.b
        
        # Original weighting from SphereFace2
        weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)
        weight = self.lw * self.num_class / self.r * weight
        
        # Calculate probabilities (sigmoid output)
        probs = torch.sigmoid(logits)
        
        # Calculate pt (probability of true class)
        pt = one_hot * probs + (1 - one_hot) * (1 - probs)
        
        # Apply focal loss modulator: (1-pt)^gamma
        focal_weight = (1 - pt).pow(self.focal_gamma)
        
        # Combine focal weighting with original weighting and detach to avoid backprop through weight
        combined_weight = (weight * focal_weight).detach()
        
        # Apply binary cross entropy with the combined weights
        loss = F.binary_cross_entropy_with_logits(logits, one_hot, weight=combined_weight)
                
        return loss
