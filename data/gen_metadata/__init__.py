from .mvtec import MVTecSolver
from .visa import VisASolver

GEN_DATA_SOLVER = {
    'mvtec': MVTecSolver,
    'visa': VisASolver,
}