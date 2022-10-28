# Practice force class Alexander Batchelor

from abc import ABC, abstractmethod
from ParticleS import ParticleS
from math import dist, sqrt, atan
import numpy as np
import numpy.typing as npt


class AbstractForce(ABC):
    def __init__(self, p1: ParticleS = None, p2: ParticleS = None, mass: float = 0, length: float = 0, cs: float = 0,
                 vk: npt.ArrayLike = (0, 0), vw: npt.ArrayLike = (0, 0), rho: float = 1.225, Akite: float = 0):
        self.p1 = p1
        self.p2 = p2
        self.mass = mass
        self.length = length
        self.Cs = cs
        self.vk = np.array(vk)
        self.vw = np.array(vw)
        self.rho = rho
        self.Akite = Akite
        self.vr = np.array([vw[0]-vk[0], vw[1]-vk[1]])

    @abstractmethod
    def magnitude(self):
        return

    @abstractmethod
    def orientation(self):
        return

    # @property
    # def particle_id(self):
    #     return self.p1.id


class ForceG(AbstractForce):  # Gravitational force
    def __str__(self):
        return f"Gravitational force of {self.magnitude():.1f}N with orientation {self.orientation()[0]:.3f}x, {self.orientation()[1]:.3f}y"

    def magnitude(self):
        return 9.81 * self.mass

    def orientation(self):
        return 0, -1


class ForceT(AbstractForce):  # Tether spring force
    def __str__(self):
        return f"Spring force of {self.magnitude():.1f}N with orientation {self.orientation()[0]:.3f}x, {self.orientation()[1]:.3f}y"

    def magnitude(self):
        coord1 = self.p1.coord
        coord2 = self.p2.coord
        if dist(coord1, coord2) >= self.length:
            l = dist(coord1, coord2)
            dl = l-self.length
            f = dl*self.Cs
            return f
        else:
            return 0

    def orientation(self):
        coord1 = self.p1.coord
        coord2 = self.p2.coord
        if dist(coord1, coord2) >= self.length:
            l = dist(coord1, coord2)
            dx = coord2[0]-coord1[0]
            dy = coord2[1]-coord1[1]
            orientation = (dx/l, dy/l)      # orientation for force acting on particle 1
            return np.array(orientation)
        else:
            return 1, 1


# class ForceD(AbstractForce):  # Tether damper force
#     def __str__(self):
#         return f"Spring force of {self.magnitude():.1f}N with orientation {self.orientation()[0]:.3f}x, {self.orientation()[1]:.3f}y"
#
#     def magnitude(self):
#         coord1 = self.p1.coord
#         coord2 = self.p2.coord
#         if dist(coord1, coord2) >= self.length:
#             l = dist(coord1, coord2)
#             dl = l-self.length
#             f = dl*self.Cs
#             return f
#         else:
#             return 0
#
#     def orientation(self):
#         coord1 = self.p1.coord
#         coord2 = self.p2.coord
#         if dist(coord1, coord2) >= self.length:
#             l = dist(coord1, coord2)
#             dx = coord2[0]-coord1[0]
#             dy = coord2[1]-coord1[1]
#             orientation = (dx/l, dy/l)      # orientation for force acting on particle 1
#             return np.array(orientation)
#         else:
#             return 1, 1

# should probably be one aerodynamic force class with drag
class ForceL(AbstractForce):  # Lift force, flat plate aerodynamic model
    def __str__(self):
        return f"Aerodynamic lift force of {self.magnitude():.1f}N with orientation " \
               f"{self.orientation()[0]:.3f}x, {self.orientation()[1]:.3f}y"

    def magnitude(self):
        a = self.aoa()
        Cl = 6e-5*a**4 - 27e-4*a**3 + 0.0299*a**2 + 0.0804*a + 0.4066
        vr = sqrt(self.vr[0]**2+self.vr[1]**2)
        L = 0.5*self.rho*vr**2*Cl*self.Akite
        return L

    def aoa(self):  # angle of attack, hardcoded fixed angle for now
        # aoa = atan(self.vk[1]/-self.vk[0])
        aoa = 10
        return aoa

    def orientation(self):
        x = self.vr[0]
        y = self.vr[1]
        l = sqrt(x**2 + y**2)
        return -y/l, x/l


class ForceD(AbstractForce):  # Drag force, flat plate aerodynamic model
    def __str__(self):
        return f"Aerodynamic drag force of {self.magnitude():.1f}N with orientation " \
               f"{self.orientation()[0]:.3f}x, {self.orientation()[1]:.3f}y"

    def magnitude(self):
        a = self.aoa()
        if -4 <= a <= 12:
            Cd = -10e-7*a**6 + 6e-6*a**5 - 10e-4*a**4 + 7e-4*a**3 - 1.1e-3*a**2 - 1.08e-2*a +0.1044
            # check equation, 2nd to last term
        elif 12 < a <= 17:
            Cd = 0.0246*a - 0.1489
        else:
            Cd = 0.27
        vr = sqrt(self.vr[0]**2+self.vr[1]**2)
        D = 0.5*self.rho*vr**2*Cd*self.Akite
        return D

    def aoa(self):  # angle of attack, hardcoded fixed angle for now
        # aoa = atan(self.vk[1]/-self.vk[0])
        aoa = 10
        return aoa

    def orientation(self):
        x = self.vr[0]
        y = self.vr[1]
        l = sqrt(x**2 + y**2)
        return x/l, y/l
