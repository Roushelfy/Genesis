"""Shared CLI options for the native graph/DPR adaptation, not CPU parity."""


def add_native_atlas_arguments(parser):
    parser.add_argument(
        "--center-mode",
        "--atlas-center-mode",
        dest="atlas_center_mode",
        choices=("farthest_point", "icosahedral"),
        default="icosahedral",
    )
    parser.add_argument(
        "--selection",
        "--atlas-selection",
        dest="atlas_selection",
        choices=("legacy", "continuous", "nearest", "narrow"),
        default="narrow",
    )
    parser.add_argument(
        "--narrow-fraction",
        "--atlas-narrow-fraction",
        dest="atlas_narrow_fraction",
        type=float,
        default=0.12,
    )


def native_atlas_options(args):
    return dict(
        atlas_center_mode=args.atlas_center_mode,
        atlas_selection=args.atlas_selection,
        atlas_narrow_fraction=args.atlas_narrow_fraction,
    )


def validate_native_atlas_arguments(parser, args, regions):
    if regions and args.atlas_center_mode == "icosahedral" and (regions != 12 or args.atlas_selection == "legacy"):
        parser.error("Native icosahedral adaptation requires 12 regions and continuous/nearest/narrow selection.")
    if not 0 < args.atlas_narrow_fraction <= 1:
        parser.error("narrow-fraction must lie in (0,1].")


def weighted_resultant_error(np, actual, expected, area, *, tensor):
    """Untimed native-resultant norm: body symmetric6 or body vector3, per env."""
    multiplicity = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]) if tensor else np.ones(actual.shape[-1])
    difference = np.sum((actual - expected) ** 2 * multiplicity, axis=-1)
    reference = np.sum(expected**2 * multiplicity, axis=-1)
    return np.sqrt(np.sum(difference * area, axis=(1, 2)) / np.maximum(np.sum(reference * area, axis=(1, 2)), 1e-60))
