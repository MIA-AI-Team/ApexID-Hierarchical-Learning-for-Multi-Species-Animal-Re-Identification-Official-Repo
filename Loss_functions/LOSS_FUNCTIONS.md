# Loss Functions - Full Paper Equations

This file keeps equations in plain text so they render cleanly everywhere.

---

## 1) ArcFace (Deng et al.)

Paper form (ArcFace Eq. 3):

L3 = -(1/m_batch) * sum_{i=1..m_batch}
     log(
       exp( s * cos(theta_{y_i} + v) ) /
       ( exp( s * cos(theta_{y_i} + v) ) + sum_{j=1, j!=y_i..n} exp( s * cos(theta_j) ) )
     )

Where:
- m_batch: batch size
- n: number of classes
- s: scale
- v: additive angular margin (often denoted m in implementations)

Equivalent per-sample form used in most implementations:

L = -log( exp( s * cos(theta_y + m) ) /
          ( exp( s * cos(theta_y + m) ) + sum_{j!=y} exp( s * cos(theta_j) ) ) )

Target logit identity used in code:

cos(theta_y + m) = cos(theta_y)cos(m) - sin(theta_y)sin(m)

---

## 2) SphereFace2 (Wen et al., ICLR 2022)

Final loss from paper (Eq. 5):

L = (lambda/r) * log( 1 + exp( -r*( g(cos(theta_y)) - m ) - b ) )
  + ((1-lambda)/r) * sum_{i!=y}
      log( 1 + exp( r*( g(cos(theta_i)) + m ) + b ) )

Similarity adjustment from paper:

g(z) = 2 * ((z + 1)/2)^t - 1

Paper hyperparameters:
- lambda: positive/negative balance
- r: hard-sample mining / sharpness factor
- m: angular margin (with m_p = m_n = m in final simplified form)
- b: shared bias term (learnable threshold)
- t: similarity-adjustment power

Notes:
- This is a binary one-vs-all summed loss, not softmax CE.
- In code, this loss returns a scalar directly.

---

## 3) Dyn-arcFace (Jiao et al.)

Main dynamic ArcFace form from paper (Eq. 5 style):

L5 = -(1/m_batch) * sum_{i=1..m_batch}
     log(
       exp( s * cos( theta_{y_i} + phi(W_n, W_{y_i}) ) ) /
       ( exp( s * cos( theta_{y_i} + phi(W_n, W_{y_i}) ) )
         + sum_{j=1, j!=y_i..n} exp( s * cos(theta_j) ) )
     )

Dynamic margin function from paper (Eq. 6 + Eq. 7):

phi(W_n, W_{y_i}) = relu( phi_{y_i-n} - k1 ) * k2 + k3 + phi_smooth

phi_smooth = ( 0.03 / ( ( abs(phi_{y_i-n} - k1) / 0.05 )^1.1 + 1 ) ) * k3

Where:
- phi_{y_i-n}: angle between target class center W_{y_i} and its nearest-neighbor center W_n
- k1, k2, k3: empirical hyperparameters

Notes:
- Dyn-arcFace replaces fixed margin with class-dependent dynamic margin.
- It still uses ArcFace-style softmax cross-entropy over logits.

---

## Quick Comparison

- ArcFace: fixed additive angular margin in softmax CE.
- SphereFace2: one-vs-all binary logistic terms with similarity adjustment.
- Dyn-arcFace: ArcFace-like CE but margin is dynamic from center geometry.
