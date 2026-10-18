"""Physical, unprojected stress from the pinned QIPC SNH and StVK-Hencky energies."""

from dataclasses import dataclass

import torch


def snh_parameters(young, poisson):
    mu = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    return 4 * mu / 3, lame + 5 * mu / 6


def snh_energy(F, mu, lame):
    squared = (F * F).sum(dim=(-1, -2))
    volume = torch.linalg.det(F)
    alpha = 1 + 3 * mu / (4 * lame)
    return 0.5 * mu * (squared - 3 - torch.log(squared + 1)) + 0.5 * lame * (volume - alpha) ** 2


def snh_cauchy(F, mu, lame):
    volume = torch.linalg.det(F)
    squared = (F * F).sum(dim=(-1, -2))
    shear = mu * (1 - 1 / (squared + 1)) / volume
    pressure = lame * (volume - 1 - 3 * mu / (4 * lame))
    identity = torch.eye(3, dtype=F.dtype, device=F.device)
    return shear[..., None, None] * (F @ F.transpose(-1, -2)) + pressure[..., None, None] * identity


def hencky_cauchy(F, mu, lame):
    left, singular, _ = torch.linalg.svd(F)
    strain = torch.log(singular)
    principal = 2 * mu[..., None] * strain + lame[..., None] * strain.sum(dim=-1, keepdim=True)
    return (left * principal[..., None, :]) @ left.transpose(-1, -2) / torch.linalg.det(F)[..., None, None]


@dataclass
class SolidField:
    deformation_gradient: torch.Tensor
    determinant: torch.Tensor
    first_piola_pa: torch.Tensor
    cauchy_pa: torch.Tensor
    von_mises_pa: torch.Tensor
    maximum_principal_pa: torch.Tensor
    energy_density_relative_rest_j_per_m3: torch.Tensor


def evaluate(F, mu, lame, model="snh"):
    determinant = torch.linalg.det(F)
    if model == "snh":
        tensor = snh_cauchy(F, mu, lame)
        rest_energy = -0.5 * mu * torch.log(F.new_tensor(4.0)) + 9 * mu**2 / (32 * lame)
        energy = snh_energy(F, mu, lame) - rest_energy
    elif model == "hencky":
        tensor = hencky_cauchy(F, mu, lame)
        strain = torch.log(torch.linalg.svdvals(F))
        energy = mu * (strain * strain).sum(dim=-1) + 0.5 * lame * strain.sum(dim=-1) ** 2
    else:
        raise ValueError(model)
    piola = determinant[..., None, None] * tensor @ torch.linalg.inv(F).transpose(-1, -2)
    deviator = tensor - tensor.diagonal(dim1=-1, dim2=-2).sum(dim=-1)[..., None, None] / 3 * torch.eye(
        3, dtype=F.dtype, device=F.device
    )
    return SolidField(
        F,
        determinant,
        piola,
        tensor,
        torch.sqrt(1.5 * (deviator * deviator).sum(dim=(-1, -2))),
        torch.linalg.eigvalsh(tensor)[..., -1],
        energy,
    )


def nodal_elastic_gradient(field, inverse_rest, volumes, tetrahedra, vertices):
    local_free = (field.first_piola_pa @ inverse_rest.transpose(-1, -2)).transpose(-1, -2) * volumes[:, None, None]
    local = torch.cat((-local_free.sum(dim=1, keepdim=True), local_free), dim=1)
    gradient = local.new_zeros((vertices, 3))
    gradient.index_add_(0, tetrahedra.reshape(-1), local.reshape(-1, 3))
    return gradient
