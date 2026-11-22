"""Unit-conversion helpers shared by BreastMap and BeaconMap.

Convention (same as the original code): maps have shape (Nx, Ny);
axis 0 = lateral (pixel size dx, meters), axis 1 = depth (pixel size dy, meters).
"""


class GridUnits:
    """Mixin: requires self.dx and self.dy (meters)."""

    @staticmethod
    def microns2meters(thickness):
        return thickness * 1e-6

    @staticmethod
    def meters2microns(thickness):
        return thickness * 1e6

    def microns2pixels(self, thickness):
        # round (instead of truncating) so 1e-6/dy floating-point error cannot lose a pixel
        return int(round(thickness * 1e-6 / self.dy))

    def pixels2microns(self, pixels):
        return float(pixels * self.dy * 1e6)

    def pixels2meters(self, pixels):
        return float(pixels * self.dy)

    def meters2pixels(self, meters):
        return int(round(meters / self.dy))
