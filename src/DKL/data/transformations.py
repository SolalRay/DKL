import numpy as np

def transform_tanh(x, scale=5.):
    scaling_factor = scale  # To stay within the interval [-5, 5]
    result = scaling_factor * np.tanh(x)
    return result

def transform_x_abs_x(x, scale=5.):
    """Applies the x*|x| transformation with a given center"""
    offset = x
    r = np.linalg.norm(offset, axis=1, keepdims=True)
    result = offset * r
    return result

def transform_sin(points, scale=5.0, amp_x=1, freq_x=1, amp_y=2, freq_y=1):
    """
    Distortion by sinusoidal waves - compatible with both PyTorch and NumPy
    """
    # Check if the input is a PyTorch tensor or a NumPy array
    is_torch = hasattr(points, 'dtype') and 'torch' in str(points.dtype)
    
    if is_torch:
        import torch
        # PyTorch version
        x, y = points[:, 0], points[:, 1]
        x_new = x + amp_x * torch.sin(freq_x * y)
        y_new = y + amp_y * torch.sin(freq_y * x)
        return torch.stack((x_new, y_new), dim=1)
    else:
        import numpy as np
        # NumPy version
        x, y = points[:, 0], points[:, 1]
        x_new = x + amp_x * np.sin(freq_x * y)
        y_new = y + amp_y * np.sin(freq_y * x)
        return np.column_stack((x_new, y_new))

def transform_spiral(x, centre=np.array([0., 0.]), num_twists=2):
    """Transforms points by applying a spiral distortion."""
    offset = x - centre  # Offset relative to the center
    r = np.linalg.norm(offset, axis=1)  # Distance to center (shape: (num_samples,))
    theta = np.arctan2(offset[:, 1], offset[:, 0])  # Initial angle

    # Add a rotation proportional to r to create a spiral
    theta_new = theta + num_twists * np.pi * r / 4  # Spiral effect

    # New transformed points
    result = np.stack([r * np.cos(theta_new), r * np.sin(theta_new)], axis=1) + centre
    return result

def injective_multi_spiral(z, num_centers=3, twist_strength=2.0, influence_radius=0.8):
    """
    Continuous injective transformation with multiple vortices.
    
    Parameters:
    - z : (N, 2) uniform samples ∈ [-1, 1]^2
    - num_centers : number of rotation centers
    - twist_strength : rotation strength
    - influence_radius : influence radius of a vortex

    Returns:
    - x : (N, 2) transformed points
    """
    x = z.copy()

    # Generate centers distributed in a circle
    angles = np.linspace(0, 2 * np.pi, num_centers, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 0.8  # placed within [-1, 1]^2

    for center in centers:
        offset = x - center  # center -> points vectors
        dist = np.linalg.norm(offset, axis=1, keepdims=True)

        # Radial attenuation function (avoids multiple interactions)
        mask = np.exp(-(dist**2) / (2 * influence_radius**2))  # ∈ [0,1]

        # Orthogonal rotation of the offset vector (90° in the plane)
        rotated = np.stack([-offset[:, 1], offset[:, 0]], axis=1)

        # Add the weighted rotation field
        x += twist_strength * mask * rotated

    return x