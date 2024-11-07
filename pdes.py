def pde_loss(pred, true, B, x):

    # Extract the specific features (1st and 3rd) for n and v_parallel

    # Plasma density & ion velocity element 1 en 3 variabele 
    pred_n = pred[:, :, :, 0]  # Shape: [batch_size, time_steps, 500]2,2,500
    pred_v_parallel = pred[:, :, :, 2]  # Shape: [batch_size, time_steps, 500]
    true_n = true[:, :, :, 0]  # Shape: [batch_size, time_steps, 500]2,2,500
    true_v_parallel = true[:, :, :, 2]  # Shape: [batch_size, time_steps, 500]

    # print(f"pred_n shape: {pred_n.shape}")
    # print(f"pred_v_parallel shape: {pred_v_parallel.shape}")
    # print(f"B shape: {B.shape}")

    # Expand B to match the shape for broadcasting
    B = B.view(1, 1, 500).expand_as(pred_n)  # Shape: [batch_size, time_steps, 500]

    # Calculate n * v_parallel / B for both predicted and true values
    pred_n_v_parallel = (pred_n * pred_v_parallel) / B
    true_n_v_parallel = (true_n * true_v_parallel) / B

    # Initialize total loss
    total_loss = 0
    delta_x = x[1] - x[0]  # Assuming uniform spacing in x
    for sample_idx in range(pred_n.shape[0]):  # Loop over batch
        for t in range(pred_n.shape[1]):  # Loop over timesteps
          # Use finite difference approximation for spatial derivative of predicted values
          pred_nv = pred_n_v_parallel[sample_idx, t, :]
          d_pred_nv_dx = (pred_nv[1:] - pred_nv[:-1]) / delta_x

          # Use finite difference approximation for spatial derivative of ground truth values
          true_nv = true_n_v_parallel[sample_idx, t, :]
          d_true_nv_dx = (true_nv[1:] - true_nv[:-1]) / delta_x

          # Calculate the squared difference for the spatial component
          spatial_loss = B[sample_idx, t, 1:] ** 2 * (d_pred_nv_dx - d_true_nv_dx) ** 2

          # Accumulate the spatial loss
          total_loss += spatial_loss.mean()
    # Return the mean loss over all timesteps
    f_n = total_loss / pred_n.shape[1]
    return f_n

def finite_difference(arr, delta):
    """Compute finite difference for a 1D tensor."""
    return (arr[1:] - arr[:-1]) / delta