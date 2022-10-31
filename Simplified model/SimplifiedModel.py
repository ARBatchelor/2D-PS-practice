"""
OO simplified practice PS model
Alexander Batchelor, TU Delft, Msc SET track
Assumptions include:     2D model, flat plate aerodynamics, point mass kite,
                         uniform static wind field, single tether segment, fixed angle of attack ????
"""

from ParticleS import ParticleS
from AbstractForce import ForceDampT, ForceG, ForceT, ForceL, ForceD, ForceDampA
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from math import sqrt
# from NumericalIntegrationS import impl_euler

# all kite parameters are just guesstimates for now
class SimplifiedModel:
    def __init__(self, pos: npt.ArrayLike = (20, 40), gs: npt.ArrayLike = (0, 0), mass: float = 10, tether_l: float = 60,
                 vw: npt.ArrayLike = (5, 0), vk0: npt.ArrayLike = (-3, 1), Akite: float = 10):
        # model parameters
        self.pos = np.array(pos)        # kite particle coordinates [m]
        self.gs = np.array(gs)          # ground station particle coordinates [m]
        self.mass = mass                # kite mass [kg]
        self.tether_l = tether_l        # tether length [m]
        self.vw = np.array(vw)          # wind speed vector of static wind field [m/s]
        self.vk = np.array(vk0)         # initial kite velocity vector [m/s]
        self.Akite = Akite              # area kite [m2]

        # pre allocating attributes
        self.particles = {}
        self.forces = {}
        self.cs = 300                    # spring coefficient [N/m]
        self.cd = 100                    # tether damping coefficient [N/m/s]
        self.cda = 20                    # aerodynamic damping coefficient [N/m/s]
        self.resultant = np.array([0, 0])
        self.acc = np.array([0, 0])      # acceleration vector

        # simulation parameters
        self.numintmethod = 1
        self.t = 0
        self.dt = 0.01
        self.T = 30

        # instantiate particles and forces
        self.update()
        self.calc_resultant()

    def update(self):
        # Particles
        if self.particles == {}:
            self.particles["gs_p"] = self.particle(self.gs)
            self.particles["kite_p"] = self.particle(self.pos)
        else:
            self.particles["kite_p"].pos = self.pos

        # Forces
        if self.forces == {}:
            self.forces["gravitational"] = ForceG(self.particles["kite_p"], mass=self.mass)
            self.forces["tether"] = ForceT(self.particles["kite_p"], self.particles["gs_p"], self.mass, self.tether_l, self.cs)
            self.forces["lift"] = ForceL(self.particles["kite_p"], vw=self.vw, vk=self.vk, Akite=self.Akite)
            self.forces["drag"] = ForceD(self.particles["kite_p"], vw=self.vw, vk=self.vk, Akite=self.Akite)
            self.forces["damper"] = ForceDampT(self.particles["kite_p"], self.particles["gs_p"], vk=self.vk, cd=self.cd)
            self.forces["a_damping"] = ForceDampA(vk=self.vk, cda=self.cda)

        # need a 'smarter' way of recalculating/updating forces here, probably abstract method 
        # (for force in self.forces.items(): force.update(v(id), pos(id)) )
        else:
            self.forces["tether"] = ForceT(self.particles["kite_p"], self.particles["gs_p"], self.mass, self.tether_l, self.cs)
            self.forces["lift"] = ForceL(self.particles["kite_p"], vw=self.vw, vk=self.vk, Akite=self.Akite)
            self.forces["drag"] = ForceD(self.particles["kite_p"], vw=self.vw, vk=self.vk, Akite=self.Akite)
            self.forces["damper"] = ForceDampT(self.particles["kite_p"], self.particles["gs_p"], vk=self.vk, cd=self.cd)
            self.forces["a_damping"] = ForceDampA(vk=self.vk, cda=self.cda)

    @staticmethod
    def particle(coord: npt.ArrayLike):
        return ParticleS(coord)

    # @staticmethod
    # def force():
    #     return ForceS()
    def calc_resultant(self):
        resultant = np.array([0, 0])
        for force in self.forces.values():
            resultant = np.add(resultant, np.array([force.magnitude()*force.orientation()[0], force.magnitude()*force.orientation()[1]]))
        self.resultant = resultant

    def propagate(self):        # probably separate file for choice of propagation method later
        self.calc_resultant()

        if self.numintmethod == 1:      # Explicit Euler: y(t+dt) = f(y(t))
            self.acc = np.array([self.resultant[0] / self.mass, self.resultant[1] / self.mass])
            self.pos = np.add(self.pos, self.vk*self.dt)
            self.vk = np.add(self.vk, self.acc*self.dt)
            self.update()
        elif self.numintmethod == 2:    # Implicit Euler: y(t+dt) = f(y(t), y(t+dt))
            dv = impl_euler(self.pos, self.vk, self.resultant, self.dt, self.mass)

            return

    def plot(self):             # probably separate file for visualization later
        fig = plt.figure(figsize=(10, 10), dpi=80)
        ax = fig.add_subplot(111)
        initial_resultant = self.resultant
        while self.t < self.T:

            plt.cla()
            x = []
            y = []
            dx = []
            dy = []
            for name, particle in self.particles.items():
                x.append(particle.coord[0])
                y.append(particle.coord[1])

            for name, force in self.forces.items():
                dx.append(force.orientation()[0])
                dy.append(force.orientation()[1])

            plt.plot(x, y)
            # for j in range(0,len(x)):
            # plt.arrow(x[1], y[1], dx[0], dy[0], width=0.05, color='red')
            # plt.arrow(x[1], y[1], dx[1], dy[1], width=0.05, color='red')
            # orientation = self.forces["damper"].orientation()
            # plt.arrow(self.pos[0], self.pos[1], orientation[0], orientation[1], width=0.05, color='red')
            sizing = sqrt((self.resultant[0]**2+self.resultant[1]**2)/(initial_resultant[0]**2 + initial_resultant[1]**2))
            # print(sizing)
            orientation = self.resultant/sqrt(self.resultant[0]**2+self.resultant[1]**2)  # *sizing
            plt.arrow(self.pos[0], self.pos[1], orientation[0], orientation[1], width=0.05, color='red')

            for i in range(0, len(x)):
                plt.plot(x[i], y[i], 'go')

            plt.ylabel("Height [m]")
            plt.xlabel("Horizontal distance from ground station in direction of wind field [m]")
            ax.set(xlim=(-2, 25), ylim=(-2, 80))
            ax.set_aspect('equal', 'box')
            ax.set_xticks([0, 5, 10, 20])
            ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
            self.propagate()
            # plt.pause(1)
            plt.pause(self.dt/2)
            self.t += self.dt

        plt.show()


if __name__ == '__main__':
    model = SimplifiedModel()
    model.plot()

    # for name, force in model.forces.items():
    #     print(name, f"{force.magnitude():.1f} N", force.orientation())

    for i in range(100):
        model.propagate()
        # if model.resultant >= 10000:
        for name, force in model.forces.items():
            print(name, f"{force.magnitude():.1f} N", force.orientation())
        print(f"vk: {model.vk}, Fres: {model.resultant} N")
        print(f"pos: {model.pos}")
        print()