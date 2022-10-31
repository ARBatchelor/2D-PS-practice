# Practice particle class Alexander Batchelor
import numpy as np
import numpy.typing as npt


class ParticleS:
    def __init__(self, pos: npt.ArrayLike):
        self.pos = np.array(pos)

    def __str__(self):
        return f"Particle Object with coordinates [{self.pos[0]}, {self.pos[1]}]"

    @property
    def coord(self):
        return self.pos[0], self.pos[1]

    # @property
    # def id(self):
    #     return self.id
