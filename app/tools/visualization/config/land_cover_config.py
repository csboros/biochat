"""Configuration for land cover types and colors."""

class LandCoverConfig:
    """Configuration class for land cover types and colors."""

    # Define the land cover classes with their descriptions
    LAND_COVER_CLASSES = {
        0: "Unknown.",
        20: "Shrubland.",
        30: "Herbaceous vegetation.",
        40: "Cropland.",
        50: "Urban/built up.",
        60: "Bare/sparse vegetation.",
        70: "Snow and ice.",
        80: "Permanent water bodies.",
        90: "Herbaceous wetland.",
        100: "Moss and lichen.",
        111: "Closed forest, evergreen needle leaf.",
        112: "Closed forest, evergreen broad leaf.",
        113: "Closed forest, deciduous needle leaf.",
        114: "Closed forest, deciduous broad leaf.",
        115: "Closed forest, mixed.",
        116: "Closed forest, not matching any of the other definitions.",
        121: "Open forest, evergreen needle leaf.",
        122: "Open forest, evergreen broad leaf.",
        123: "Open forest, deciduous needle leaf.",
        124: "Open forest, deciduous broad leaf.",
        125: "Open forest, mixed.",
        126: "Open forest, not matching any of the other definitions.",
        200: "Oceans, seas."
    }

    # Define the color palette
    COLOR_PALETTE = [
        '#282828',  # Unknown
        '#ffbb22',  # Shrubs
        '#ffff4c',  # Herbaceous vegetation
        '#f096ff',  # Cultivated and managed vegetation
        '#fa0000',  # Urban / built up
        '#b4b4b4',  # Bare / sparse vegetation
        '#f0f0f0',  # Snow and ice
        '#0032c8',  # Permanent water bodies
        '#0096a0',  # Herbaceous wetland
        '#fae6a0',  # Moss and lichen
        '#58481f',  # Closed forest, evergreen needle leaf
        '#009900',  # Closed forest, evergreen broad leaf
        '#70663e',  # Closed forest, deciduous needle leaf
        '#00cc00',  # Closed forest, deciduous broad leaf
        '#4e751f',  # Closed forest, mixed
        '#007800',  # Closed forest, not matching other definitions
        '#666000',  # Open forest, evergreen needle leaf
        '#8db400',  # Open forest, evergreen broad leaf
        '#8d7400',  # Open forest, deciduous needle leaf
        '#a0dc00',  # Open forest, deciduous broad leaf
        '#929900',  # Open forest, mixed
        '#648c00',  # Open forest, not matching other definitions
        '#000080'   # Oceans, seas
    ]

    # Create a mapping of codes to colors
    CODE_TO_COLOR = {
        0: COLOR_PALETTE[0],    # Unknown
        20: COLOR_PALETTE[1],   # Shrubs
        30: COLOR_PALETTE[2],   # Herbaceous vegetation
        40: COLOR_PALETTE[3],   # Cultivated and managed vegetation
        50: COLOR_PALETTE[4],   # Urban / built up
        60: COLOR_PALETTE[5],   # Bare / sparse vegetation
        70: COLOR_PALETTE[6],   # Snow and ice
        80: COLOR_PALETTE[7],   # Permanent water bodies
        90: COLOR_PALETTE[8],   # Herbaceous wetland
        100: COLOR_PALETTE[9],  # Moss and lichen
        111: COLOR_PALETTE[10], # Closed forest, evergreen needle leaf
        112: COLOR_PALETTE[11], # Closed forest, evergreen broad leaf
        113: COLOR_PALETTE[12], # Closed forest, deciduous needle leaf
        114: COLOR_PALETTE[13], # Closed forest, deciduous broad leaf
        115: COLOR_PALETTE[14], # Closed forest, mixed
        116: COLOR_PALETTE[15], # Closed forest, not matching other definitions
        121: COLOR_PALETTE[16], # Open forest, evergreen needle leaf
        122: COLOR_PALETTE[17], # Open forest, evergreen broad leaf
        123: COLOR_PALETTE[18], # Open forest, deciduous needle leaf
        124: COLOR_PALETTE[19], # Open forest, deciduous broad leaf
        125: COLOR_PALETTE[20], # Open forest, mixed
        126: COLOR_PALETTE[21], # Open forest, not matching other definitions
        200: COLOR_PALETTE[22]  # Oceans, seas
    }

    # Define visualization parameters for Earth Engine
    VIS_PARAMS = {
        'min': 0,
        'max': 200,
        'palette': [
            CODE_TO_COLOR[0],    # Unknown
            CODE_TO_COLOR[20],   # Shrubs
            CODE_TO_COLOR[30],   # Herbaceous vegetation
            CODE_TO_COLOR[40],   # Cultivated and managed vegetation
            CODE_TO_COLOR[50],   # Urban / built up
            CODE_TO_COLOR[60],   # Bare / sparse vegetation
            CODE_TO_COLOR[70],   # Snow and ice
            CODE_TO_COLOR[80],   # Permanent water bodies
            CODE_TO_COLOR[90],   # Herbaceous wetland
            CODE_TO_COLOR[100],  # Moss and lichen
            CODE_TO_COLOR[111],  # Closed forest, evergreen needle leaf
            CODE_TO_COLOR[112],  # Closed forest, evergreen broad leaf
            CODE_TO_COLOR[113],  # Closed forest, deciduous needle leaf
            CODE_TO_COLOR[114],  # Closed forest, deciduous broad leaf
            CODE_TO_COLOR[115],  # Closed forest, mixed
            CODE_TO_COLOR[116],  # Closed forest, not matching other definitions
            CODE_TO_COLOR[121],  # Open forest, evergreen needle leaf
            CODE_TO_COLOR[122],  # Open forest, evergreen broad leaf
            CODE_TO_COLOR[123],  # Open forest, deciduous needle leaf
            CODE_TO_COLOR[124],  # Open forest, deciduous broad leaf
            CODE_TO_COLOR[125],  # Open forest, mixed
            CODE_TO_COLOR[126],  # Open forest, not matching other definitions
            CODE_TO_COLOR[200]   # Oceans, seas
        ]
    }

    @classmethod
    def get_color_for_code(cls, code: int) -> str:
        """Get color for specific land cover code."""
        return cls.CODE_TO_COLOR.get(code, cls.COLOR_PALETTE[0])  # Default to unknown color

    @classmethod
    def get_vis_params(cls) -> dict:
        """Get visualization parameters for Earth Engine."""
        return cls.VIS_PARAMS.copy()
