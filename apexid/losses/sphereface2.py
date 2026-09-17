import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sphereface2_initial_bias(num_class, magn_type='C', alpha=0.7, r=40., m=0.4, t=3.):
    """Return the official SphereFace2 shared-bias initialization."""
    num_class = int(num_class)
    if num_class <= 1:
        raise ValueError("num_class must be greater than 1 for SphereFace2.")
    magn_type = str(magn_type).upper()
    z = float(alpha) / ((1. - float(alpha)) * (num_class - 1.))
    if magn_type == 'C':
        ay = float(r) * (2. * 0.5**float(t) - 1. - float(m))
        ai = float(r) * (2. * 0.5**float(t) - 1. + float(m))
    elif magn_type == 'A':
        theta_y = min(math.pi, math.pi / 2. + float(m))
        ay = float(r) * (2. * ((math.cos(theta_y) + 1.) / 2.)**float(t) - 1.)
        ai = float(r) * (2. * 0.5**float(t) - 1.)
    elif magn_type == 'M':
        theta_y = min(math.pi, float(m) * math.pi / 2.)
        ay = float(r) * (2. * ((math.cos(theta_y) + 1.) / 2.)**float(t) - 1.)
        ai = float(r) * (2. * 0.5**float(t) - 1.)
    else:
        raise NotImplementedError
    temp = (1. - z)**2 + 4. * z * math.exp(ay - ai)
    return math.log(2. * z) - ai - math.log(1. - z + math.sqrt(temp))


def sphereface2_logits_from_cosine(
    cosine,
    labels,
    *,
    magn_type='C',
    r=40.,
    m=0.4,
    t=3.,
    bias=0.,
):
    """Apply ApexID's exact SphereFace2 transform to external cosine scores.

    The detached margin delta preserves the straight-through behavior of the
    existing implementation.
    """
    if cosine.ndim != 2:
        raise ValueError("cosine must have shape [batch_size, num_classes].")
    labels = labels.long()
    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, labels.view(-1, 1), 1.)
    magn_type = str(magn_type).upper()
    with torch.no_grad():
        if magn_type == 'C':
            adjusted = 2. * ((cosine + 1.) / 2.).pow(float(t)) - 1.
            adjusted = adjusted - float(m) * (2. * one_hot - 1.)
        elif magn_type == 'A':
            theta_m = torch.acos(cosine.clamp(-1 + 1e-5, 1. - 1e-5))
            idx = labels.view(-1, 1)
            theta_m.scatter_(1, idx, theta_m.gather(1, idx) + float(m))
            theta_m.clamp_(1e-5, 3.14159)
            adjusted = torch.cos(theta_m)
            adjusted = 2. * ((adjusted + 1.) / 2.).pow(float(t)) - 1.
        elif magn_type == 'M':
            m_theta = torch.acos(cosine.clamp(-1 + 1e-5, 1. - 1e-5))
            idx = labels.view(-1, 1)
            m_theta.scatter_(1, idx, m_theta.gather(1, idx) * float(m))
            m_theta.clamp_(1e-5, 3.14159)
            adjusted = torch.cos(m_theta)
            adjusted = 2. * ((adjusted + 1.) / 2.).pow(float(t)) - 1.
        else:
            raise NotImplementedError
        delta = adjusted - cosine
    return float(r) * (cosine + delta) + bias


def sphereface2_loss_from_cosine(
    cosine,
    labels,
    *,
    magn_type='C',
    alpha=0.7,
    r=40.,
    m=0.4,
    t=3.,
    lw=50.,
    bias=0.,
    normalize_negatives=False,
):
    """Compute SphereFace2 from externally supplied per-class cosine scores.

    The default reduction is numerically identical to ordinary SphereFace2.
    BHCA's conditional branch normalizes sibling negatives per anchor.
    """
    labels = labels.long()
    logits = sphereface2_logits_from_cosine(
        cosine,
        labels,
        magn_type=magn_type,
        r=r,
        m=m,
        t=t,
        bias=bias,
    )
    targets = torch.zeros_like(cosine)
    targets.scatter_(1, labels.view(-1, 1), 1.)
    if not normalize_negatives:
        weight = float(alpha) * targets + (1. - float(alpha)) * (1. - targets)
        weight = float(lw) * cosine.shape[1] / float(r) * weight
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weight)

    element_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    positive = (element_loss * targets).sum(dim=1)
    negative_mask = 1. - targets
    negative = (element_loss * negative_mask).sum(dim=1) / negative_mask.sum(dim=1).clamp_min(1.)
    per_anchor = float(lw) / float(r) * (
        float(alpha) * positive + (1. - float(alpha)) * negative
    )
    return per_anchor.mean()


class SphereFace2(nn.Module):
    """
    SphereFace2 classification head.

    Reference:
        "SphereFace2: Binary Classification is All You Need for Deep Face Recognition"

    Main idea
    ---------
    SphereFace2 reformulates identity classification as K binary classification
    problems, where K is the number of training identities/classes.

    Instead of training with a normal softmax classifier:

        "Which identity is this image?"

    SphereFace2 trains the model using one-vs-all binary classification:

        "Is this image identity 1 or not?"
        "Is this image identity 2 or not?"
        ...
        "Is this image identity K or not?"

    This is more aligned with open-set face recognition / animal re-identification,
    because verification and re-ID are naturally based on pairwise similarity:

        same identity vs different identity

    Input/output shapes
    -------------------
    Args to forward:
        x:
            Embedding tensor from the backbone.

            Shape:
                [batch_size, feat_dim]

            Example:
                [64, 512]

        y:
            Integer class labels.

            Shape:
                [batch_size]

            Example:
                [64]

    Internal classifier weight:
        self.w:
            Trainable class proxy matrix.

            Shape:
                [feat_dim, num_class]

            Each column self.w[:, c] represents the proxy vector of class c.

    Output:
        loss:
            Scalar binary cross-entropy loss over all K binary classifiers.

    Hyperspherical classification
    -----------------------------
    SphereFace2 normalizes both:

        1. Feature embeddings x
        2. Classifier weights w

    Then the class score is computed as cosine similarity:

        cos(theta_i) = normalized_x dot normalized_w_i

    Therefore, the model learns an angular embedding space where:

        same identity      -> small angle / high cosine similarity
        different identity -> large angle / low cosine similarity

    Similarity adjustment function
    ------------------------------
    SphereFace2 applies a nonlinear similarity mapping:

        g(z) = 2 * ((z + 1) / 2)^t - 1

    where:

        z = cos(theta)

    The hyperparameter t controls how strongly the cosine similarity is reshaped.
    With the default t=3, high similarities are emphasized more strongly.

    Margin variants
    ---------------
    The argument `magn_type` selects the SphereFace2 margin type.

    1. magn_type = 'C'  -> SphereFace2-C

        This is the cosine-margin version.

        Positive class:
            g(cos(theta_y)) - m

        Negative classes:
            g(cos(theta_i)) + m

        So the true class is made harder by subtracting margin m,
        and negative classes are made harder by adding margin m.

    2. magn_type = 'A'  -> SphereFace2-A

        This is the additive angular-margin version.

        For the positive class only:

            theta_y -> theta_y + m

        Then the adjusted angle is converted back to cosine and passed through g.

    3. magn_type = 'M'  -> SphereFace2-M

        This is the multiplicative angular-margin version.

        For the positive class only:

            theta_y -> m * theta_y

        Then the adjusted angle is converted back to cosine and passed through g.

    Important implementation detail
    -------------------------------
    The margin adjustment is computed inside `torch.no_grad()`.

    This means:
        - the forward logits use the margin-adjusted value
        - the backward gradient mostly flows through the original cosine similarity

    This acts like a straight-through margin adjustment and matches the original
    OpenSphere implementation style.

    Hyperparameters
    ---------------
    feat_dim:
        Dimension of the embedding vector produced by the backbone.

    num_class:
        Number of training identities/classes.

    magn_type:
        Margin type:
            'C' for SphereFace2-C
            'A' for SphereFace2-A
            'M' for SphereFace2-M

    alpha:
        Positive/negative balancing coefficient.

        This corresponds to lambda in the SphereFace2 paper.

        Positive samples receive weight:
            alpha

        Negative samples receive weight:
            1 - alpha

        Default:
            0.7

    r:
        Radius / logit scale factor.

        Larger r makes the binary classification task sharper and increases the
        effect of hard examples.

        Default:
            40.0

    m:
        Margin value.

        Its meaning depends on magn_type:

            'C': cosine-space margin
            'A': additive angular margin
            'M': multiplicative angular margin

        Default:
            0.4

    t:
        Power used in the similarity adjustment function g(z).

        Default:
            3.0

    lw:
        Extra loss-weight scaling factor used by the official implementation.

        Default:
            50.0

    Bias initialization
    -------------------
    SphereFace2 uses a learnable scalar bias b shared across all binary classifiers.

    The bias is initialized analytically so that the initial positive and negative
    binary losses are balanced according to alpha, num_class, r, m, and t.

    Training behavior
    -----------------
    For a sample with label y:

        one_hot[c] = 1 if c == y
        one_hot[c] = 0 otherwise

    BCE target for the correct class:
        1

    BCE target for all incorrect classes:
        0

    Final binary logits:

        logits = r * adjusted_cosine + b

    Final loss:

        BCEWithLogits(logits, one_hot, weight=sample_weight)

    Notes
    -----
    This head is mainly used during training to shape the embedding space.

    For open-set re-identification or verification, the usual inference output is
    the normalized embedding from the backbone, not the classifier logits.
    Similarity is usually computed with cosine similarity between embeddings.
    """

    def __init__(
        self,
        feat_dim,
        num_class,
        magn_type='C',
        alpha=0.7,
        r=40.,
        m=0.4,
        t=3.,
        lw=50.
    ):
        """
        Initialize the SphereFace2 head.

        Parameters
        ----------
        feat_dim : int
            Dimension of the input embedding vector.

            Example:
                If the backbone outputs a 512-dimensional embedding,
                then feat_dim = 512.

        num_class : int
            Number of training identities/classes.

            Example:
                If the dataset contains 1000 known individuals,
                then num_class = 1000.

        magn_type : str, default='C'
            Type of SphereFace2 margin.

            Supported values:
                'C' -> cosine-margin SphereFace2-C
                'A' -> additive angular-margin SphereFace2-A
                'M' -> multiplicative angular-margin SphereFace2-M

        alpha : float, default=0.7
            Positive/negative loss balancing coefficient.

            In the paper, this is lambda.

            Positive-class binary losses are weighted by:
                alpha

            Negative-class binary losses are weighted by:
                1 - alpha

        r : float, default=40.0
            Logit scale / radius parameter.

            The adjusted cosine score is multiplied by r before BCEWithLogits.

        m : float, default=0.4
            Margin value.

            For magn_type='C':
                margin is applied in cosine space.

            For magn_type='A':
                margin is added to the positive angle.

            For magn_type='M':
                positive angle is multiplied by the margin.

        t : float, default=3.0
            Power parameter in the nonlinear similarity transformation:

                g(z) = 2 * ((z + 1) / 2)^t - 1

        lw : float, default=50.0
            Extra loss scaling factor used in the original implementation.
        """

        super().__init__()

        self.feat_dim = feat_dim
        self.num_class = num_class
        self.magn_type = magn_type

        # alpha is lambda in the SphereFace2 paper.
        # It balances positive and negative binary classification losses.
        self.alpha = alpha

        # r is the radius / logit scale factor.
        self.r = r

        # m is the margin.
        self.m = m

        # t controls the nonlinear similarity transformation g(z).
        self.t = t

        # lw is an additional loss scaling factor used by the official code.
        self.lw = lw

        # Trainable class proxy matrix.
        #
        # Shape:
        #   [feat_dim, num_class]
        #
        # Each column represents one class/identity proxy.
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

        # ------------------------------------------------------------
        # Bias initialization
        # ------------------------------------------------------------
        #
        # SphereFace2 uses a shared learnable scalar bias b.
        #
        # This initialization is derived so that the initial positive
        # and negative losses are balanced according to:
        #
        #   alpha, num_class, r, m, and t.
        #
        # z controls the desired positive/negative weighting ratio.
        z = alpha / ((1. - alpha) * (num_class - 1.))

        if magn_type == 'C':
            # SphereFace2-C:
            #
            # At initialization, assume cos(theta) = 0.
            #
            # g(0) = 2 * (0.5)^t - 1
            #
            # Positive adjusted score:
            #   g(0) - m
            #
            # Negative adjusted score:
            #   g(0) + m
            ay = r * (2. * 0.5**t - 1. - m)
            ai = r * (2. * 0.5**t - 1. + m)

        elif magn_type == 'A':
            # SphereFace2-A:
            #
            # Additive angular margin.
            #
            # For the positive class:
            #   theta_y = pi/2 + m
            #
            # Then convert back to cosine and apply g.
            theta_y = min(math.pi, math.pi / 2. + m)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.)**t - 1.)

            # Negative classes use theta = pi/2, so cos(theta)=0.
            ai = r * (2. * 0.5**t - 1.)

        elif magn_type == 'M':
            # SphereFace2-M:
            #
            # Multiplicative angular margin.
            #
            # For the positive class:
            #   theta_y = m * pi/2
            #
            # Then convert back to cosine and apply g.
            theta_y = min(math.pi, m * math.pi / 2.)
            ay = r * (2. * ((math.cos(theta_y) + 1.) / 2.)**t - 1.)

            # Negative classes use theta = pi/2, so cos(theta)=0.
            ai = r * (2. * 0.5**t - 1.)

        else:
            raise NotImplementedError

        # Closed-form initialization for shared bias b.
        temp = (1. - z)**2 + 4. * z * math.exp(ay - ai)
        b = (
            math.log(2. * z)
            - ai
            - math.log(1. - z + math.sqrt(temp))
        )

        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)

    def forward(self, x, y):
        """
        Compute the SphereFace2 training loss.

        Parameters
        ----------
        x : torch.Tensor
            Backbone embeddings.

            Shape:
                [batch_size, feat_dim]

            The embeddings are normalized internally before computing cosine
            similarities with class proxies.

        y : torch.Tensor
            Ground-truth class labels.

            Shape:
                [batch_size]

            Each value should be an integer in:

                [0, num_class - 1]

        Returns
        -------
        loss : torch.Tensor
            Scalar SphereFace2 loss.

        Step-by-step computation
        ------------------------
        1. Normalize classifier weights

            self.w is normalized along dim=0.

            Since self.w has shape:

                [feat_dim, num_class]

            normalizing along dim=0 means every class proxy column has unit norm.

        2. Compute cosine similarities

            x is normalized along dim=1.

            cos_theta = normalized_x @ normalized_w

            Shape:
                [batch_size, num_class]

        3. Build one-hot binary targets

            For each sample:

                target for true class = 1
                target for all other classes = 0

        4. Apply SphereFace2 margin adjustment

            Depending on magn_type:

                'C':
                    apply cosine margin:
                        positive: g(cos_theta) - m
                        negative: g(cos_theta) + m

                'A':
                    add angular margin to the positive class angle:
                        theta_y = theta_y + m

                'M':
                    multiply the positive class angle:
                        theta_y = m * theta_y

        5. Compute detached difference

            d_theta = adjusted_score - original_cos_theta

            This is computed under torch.no_grad().

            Then:

                cos_theta + d_theta

            has the adjusted forward value, while gradients mostly flow through
            the original cos_theta.

        6. Compute binary logits

            logits = r * adjusted_score + b

        7. Compute weighted BCE loss

            Positive class weight:
                alpha

            Negative class weight:
                1 - alpha

            Then all weights are scaled by:

                lw * num_class / r
        """

        # Keep classifier proxies normalized.
        #
        # Shape of self.w:
        #   [feat_dim, num_class]
        #
        # dim=0 means each class proxy column is normalized independently.
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # Normalize input embeddings and compute cosine similarity
        # with every class proxy.
        #
        # F.normalize(x, dim=1):
        #   [batch_size, feat_dim]
        #
        # self.w:
        #   [feat_dim, num_class]
        #
        # cos_theta:
        #   [batch_size, num_class]
        cos_theta = F.normalize(x, dim=1).mm(self.w)

        return sphereface2_loss_from_cosine(
            cos_theta,
            y,
            magn_type=self.magn_type,
            alpha=self.alpha,
            r=self.r,
            m=self.m,
            t=self.t,
            lw=self.lw,
            bias=self.b,
        )
