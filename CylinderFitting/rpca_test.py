import numpy as np

import rpca.ealm
import rpca.ialm

RNG = np.random.default_rng()
D = RNG.random((20, 20))
A0, E0 = rpca.ealm.fit(D)
A1, E1 = rpca.ialm.fit(D)

print(A0, A1)